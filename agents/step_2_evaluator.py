"""
Step 2: Question Evaluator (Group-Based Scoring Only)
======================================================
This module implements:
1. Similarity-based filtering to remove duplicate questions across batches
2. Group-based scoring system (no pairwise comparisons)
3. Multi-model evaluation support
"""

import sys
import os
import uuid
import asyncio
import numpy as np
from typing import List, Dict, Any
from pydantic import BaseModel, Field
from sklearn.metrics.pairwise import cosine_similarity
import random
import logging
from datetime import datetime
import time
from structai import load_file, save_file

sys.path.append('.')
from utils.langchain_agent import Agent
from utils.langchain_tools import web_search
from utils.langchain_utils import CustomOpenAIEmbeddings


# ==================== Logger Setup ====================

def setup_logger(output_dir: str = "./logs") -> logging.Logger:
    """Set up a file-only logger for agent outputs."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"agent_{timestamp}.log")

    logger = logging.getLogger("agent_logger")
    logger.setLevel(logging.INFO)

    # Clear existing handlers
    logger.handlers = []

    # File handler only (no console output for agent logs)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    print(f"Agent logs will be saved to: {log_file}")
    return logger


def get_unique_id() -> str:
    """Generate a unique identifier."""
    return uuid.uuid4().hex


# ==================== Pydantic Models ====================

class QuestionScores(BaseModel):
    """Scores for a single question on three dimensions."""
    novelty: float = Field(description="Novelty score (0-10)", ge=0, le=10)
    feasibility: float = Field(description="Feasibility score (0-10)", ge=0, le=10)
    significance: float = Field(description="Significance score (0-10)", ge=0, le=10)


class BatchScores(BaseModel):
    """Scores for a batch of questions scored together."""
    scores: List[QuestionScores] = Field(description="List of scores for each question")
    reasoning: str = Field(description="Overall reasoning for the scoring")


# ==================== Similarity Filter ====================

class IncrementalSimilarityFilter:
    """
    Incremental similarity filter using a representative-set strategy.

    Embeddings are computed for each incoming batch and compared against
    previously kept embeddings. Questions that exceed the similarity threshold
    are discarded as near-duplicates.
    """

    def __init__(self, similarity_threshold: float = 0.85, batch_size: int = 50):
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        self.representative_embeddings = []
        self.filtered_questions = []

        self.embeddings = CustomOpenAIEmbeddings(
            model="Qwen/Qwen3-Embedding-8B",
            api_key=os.environ.get("LLM_API_KEY"),
            base_url=os.environ.get("LLM_BASE_URL", "https://api.openai.com/v1"),
            tiktoken_enabled=False,
            tiktoken_model_name="Qwen/Qwen3-Embedding-8B",
        )

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings for texts in batches to stay within model limits."""
        if not texts:
            return np.array([])

        EMBED_BATCH_SIZE = 32
        all_embeddings = []

        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch_texts = texts[i:i+EMBED_BATCH_SIZE]
            batch_embeddings = self.embeddings.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)

        return np.array(all_embeddings)

    def _compute_max_similarity(self, query_embedding: np.ndarray) -> float:
        if len(self.representative_embeddings) == 0:
            return 0.0

        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            np.array(self.representative_embeddings)
        )[0]
        return float(np.max(similarities))

    def filter_batch(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not questions:
            return []

        # Compute embeddings for all questions in the batch
        question_texts = [q['question'] for q in questions]
        batch_embeddings = self._get_embeddings(question_texts)

        # Create (original_index, question, embedding) tuples
        indexed_items = list(enumerate(zip(questions, batch_embeddings)))

        # Shuffle to process in random order (avoids positional bias)
        random.shuffle(indexed_items)

        # Track which indices to keep (preserving original order)
        kept_indices = []

        for original_idx, (_, embedding) in indexed_items:
            max_sim = self._compute_max_similarity(embedding)

            if max_sim < self.similarity_threshold:
                kept_indices.append(original_idx)
                self.representative_embeddings.append(embedding)

        # Restore original order
        kept_indices.sort()

        filtered_batch = [questions[idx] for idx in kept_indices]
        self.filtered_questions.extend(filtered_batch)

        return filtered_batch

    def get_filtered_questions(self) -> List[Dict[str, Any]]:
        return self.filtered_questions


# ==================== Group-Based Scorer ====================

class GroupBasedScorer:
    """
    Group-based scoring without pairwise comparisons.

    Each round randomly partitions questions into groups of ``group_size`` and
    scores each group with all configured models in parallel.
    """

    def __init__(self, models: List[str] = None, comparison_rounds: int = 3,
                 group_size: int = 5, logger: logging.Logger = None,
                 max_concurrent_tasks: int = 10):
        self.models = models or ["gpt-4o-mini"]
        self.comparison_rounds = comparison_rounds
        self.group_size = group_size
        self.logger = logger
        self.max_concurrent_tasks = max_concurrent_tasks
        self.semaphore = asyncio.Semaphore(max_concurrent_tasks)

    def _get_system_prompt(self) -> str:
        """Return the system prompt for scoring agents."""
        return """You are a world-class scientific expert evaluating research questions.

Assess each question based on:
1. **Novelty**: How innovative and original? Does it explore new territories or challenge paradigms?
2. **Feasibility**: Can this be investigated with current or near-future technology and methodology?
3. **Significance**: What is the potential scientific impact if answered? Would it advance the field significantly?

Use web_search tool to check the current state of research and make informed evaluations.

Be objective and provide detailed reasoning for your assessments."""

    async def _score_group(self, questions: List[Dict[str, Any]], model: str,
                          field: str = "") -> List[QuestionScores]:
        """Score a group of questions with a single model call."""
        question_parts = []
        for i, q in enumerate(questions):
            parts = [f"Question {i}:"]

            parts.append(f"Question: {q.get('question', 'N/A')}")

            if 'background' in q and q['background']:
                parts.append(f"Background: {q['background']}")

            if 'significance' in q and q['significance']:
                parts.append(f"Significance: {q['significance']}")

            if 'methodology' in q and q['methodology']:
                if isinstance(q['methodology'], list):
                    parts.append(f"Methodology: {' '.join(q['methodology'])}")
                else:
                    parts.append(f"Methodology: {q['methodology']}")

            if 'rationale' in q and q['rationale']:
                parts.append(f"Rationale: {q['rationale']}")

            if 'key_concepts' in q and q['key_concepts']:
                if isinstance(q['key_concepts'], list):
                    parts.append(f"Key Concepts: {', '.join(q['key_concepts'])}")
                else:
                    parts.append(f"Key Concepts: {q['key_concepts']}")

            question_parts.append("\n".join(parts))

        question_list = "\n\n".join(question_parts)

        context = f"Field of study: {field}\n\n" if field else ""

        prompt = f"""{context}Please evaluate the following {len(questions)} scientific research questions.

Score each question on three dimensions (0-10 scale):
- Novelty: Innovation and originality
- Feasibility: Technical achievability
- Significance: Potential scientific impact

{question_list}

You may use web_search to check current research state.

Provide scores for all questions. Consider them independently and score based on their absolute quality."""

        agent = Agent(
            name=f"GroupScorer-{model}-{get_unique_id()}",
            model_settings={"model": model},
            system_prompt=self._get_system_prompt(),
            tools=[web_search],
            response_format=BatchScores,
            verbose=True,
            max_tool_iterations=5,
            logger=self.logger,
        )

        try:
            result = await agent.chat(prompt)
            return result.scores
        except Exception as e:
            print(f"Error scoring group with {model}: {e}")
            # Fallback: neutral scores
            return [QuestionScores(novelty=5.0, feasibility=5.0, significance=5.0) for _ in questions]

    async def score_questions(self, questions: List[Dict[str, Any]],
                            field: str = "") -> Dict[str, Any]:
        """
        Score all questions using group-based evaluation with multiple rounds.

        Each round randomly partitions questions into groups and scores each group
        with all configured models in parallel.
        """
        total_start_time = time.time()
        print(f"\nScoring {len(questions)} questions with {len(self.models)} models...")
        print(f"Using {self.comparison_rounds} rounds, group size={self.group_size}")

        # Initialize score accumulation: {model: [{novelty_sum, feasibility_sum, significance_sum, count}, ...]}
        all_model_scores = {model: [
            {"novelty": 0.0, "feasibility": 0.0, "significance": 0.0, "count": 0}
            for _ in questions
        ] for model in self.models}

        print(f"\nGroup-based scoring ({self.comparison_rounds} rounds)...")

        for round_idx in range(self.comparison_rounds):
            round_start = time.time()
            print(f"\n  Round {round_idx + 1}/{self.comparison_rounds}")

            # Randomly shuffle and partition questions into groups for this round
            shuffled_indices = list(range(len(questions)))
            random.shuffle(shuffled_indices)

            groups = []
            for i in range(0, len(shuffled_indices), self.group_size):
                group_indices = shuffled_indices[i:i+self.group_size]
                group = [questions[idx] for idx in group_indices]
                groups.append((group_indices, group))

            # Build the full list of scoring tasks (all groups × all models)
            scoring_tasks = []
            for model in self.models:
                for group_idx, (indices, group) in enumerate(groups):
                    scoring_tasks.append((model, group_idx, indices, group))

            async def score_group_task(model, group_idx, indices, group):
                async with self.semaphore:
                    group_scores = await self._score_group(group, model, field)
                    return model, group_idx, indices, group_scores

            print(f"    Scoring {len(groups)} groups with {len(self.models)} models...")
            results = await asyncio.gather(*[
                score_group_task(model, group_idx, indices, group)
                for model, group_idx, indices, group in scoring_tasks
            ])

            # Accumulate scores
            for model, group_idx, indices, group_scores in results:
                for local_idx, question_idx in enumerate(indices):
                    if local_idx < len(group_scores):
                        scores = group_scores[local_idx]
                        scores_dict = scores.model_dump() if hasattr(scores, 'model_dump') else scores
                        all_model_scores[model][question_idx]["novelty"] += scores_dict.get("novelty", 5.0)
                        all_model_scores[model][question_idx]["feasibility"] += scores_dict.get("feasibility", 5.0)
                        all_model_scores[model][question_idx]["significance"] += scores_dict.get("significance", 5.0)
                        all_model_scores[model][question_idx]["count"] += 1

            round_time = time.time() - round_start
            print(f"    Round {round_idx + 1} completed in {round_time:.2f}s")

        # Aggregate scores across rounds and models
        print("\nAggregating scores across models...")
        questions_with_scores = []

        for q_idx, question in enumerate(questions):
            question_with_scores = question.copy()
            scores_by_model = {}

            for model in self.models:
                model_data = all_model_scores[model][q_idx]
                count = model_data["count"]
                if count > 0:
                    avg_scores = {
                        "novelty": model_data["novelty"] / count,
                        "feasibility": model_data["feasibility"] / count,
                        "significance": model_data["significance"] / count,
                    }
                else:
                    avg_scores = {"novelty": 5.0, "feasibility": 5.0, "significance": 5.0}

                avg_scores["count"] = count
                scores_by_model[model] = avg_scores

            question_with_scores['scores'] = scores_by_model

            # Compute overall average across all models
            avg_novelty = np.mean([s['novelty'] for s in scores_by_model.values()])
            avg_feasibility = np.mean([s['feasibility'] for s in scores_by_model.values()])
            avg_significance = np.mean([s['significance'] for s in scores_by_model.values()])
            avg_total = (avg_novelty + avg_feasibility + avg_significance) / 3

            question_with_scores['average_scores'] = {
                'novelty': float(avg_novelty),
                'feasibility': float(avg_feasibility),
                'significance': float(avg_significance),
                'total': float(avg_total)
            }

            questions_with_scores.append(question_with_scores)

        # Consensus ranking by total score
        consensus_ranking = sorted(
            range(len(questions_with_scores)),
            key=lambda i: questions_with_scores[i]['average_scores']['total'],
            reverse=True
        )

        for idx in consensus_ranking:
            rank_position = consensus_ranking.index(idx)
            questions_with_scores[idx]['rank'] = rank_position + 1
            questions_with_scores[idx]['rank_score'] = len(questions) - rank_position

        total_time = time.time() - total_start_time
        print(f"\nScoring complete! Total time: {total_time:.2f}s\n")

        return {
            'questions_with_scores': questions_with_scores,
            'consensus_ranking': consensus_ranking,
            'models_used': self.models
        }


# ==================== Main Evaluator ====================

class QuestionEvaluator:
    """
    Main evaluator coordinating similarity filtering and group-based scoring.
    """

    def __init__(self,
                 similarity_threshold: float = 0.85,
                 filter_batch_size: int = 50,
                 models: List[str] = None,
                 comparison_rounds: int = 3,
                 group_size: int = 5,
                 log_dir: str = "./logs",
                 max_concurrent_tasks: int = 10):
        self.similarity_threshold = similarity_threshold
        self.filter_batch_size = filter_batch_size
        self.logger = setup_logger(log_dir)
        self.filter = IncrementalSimilarityFilter(similarity_threshold, filter_batch_size)
        self.scorer = GroupBasedScorer(models, comparison_rounds, group_size,
                                      self.logger, max_concurrent_tasks)

    def load_questions(self, input_file: str) -> List[Dict[str, Any]]:
        all_questions = []

        if not os.path.exists(input_file):
            print(f"Error: Input file not found: {input_file}")
            return all_questions

        try:
            questions = load_file(input_file)
            if isinstance(questions, list):
                all_questions.extend(questions)
                print(f"Loaded {len(questions)} questions from {input_file}")
        except Exception as e:
            print(f"Error loading {input_file}: {e}")

        return all_questions

    def filter_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        start_time = time.time()
        print(f"\nFiltering {len(questions)} questions (threshold={self.similarity_threshold})...")

        num_batches = (len(questions) + self.filter_batch_size - 1) // self.filter_batch_size

        for i in range(0, len(questions), self.filter_batch_size):
            batch_start = time.time()
            batch = questions[i:i+self.filter_batch_size]
            filtered_batch = self.filter.filter_batch(batch)
            batch_time = time.time() - batch_start
            print(f"Batch {i//self.filter_batch_size + 1}/{num_batches}: "
                  f"Kept {len(filtered_batch)}/{len(batch)} questions (Time: {batch_time:.2f}s)")

        filtered_questions = self.filter.get_filtered_questions()
        total_time = time.time() - start_time
        print(f"Filtering complete: {len(filtered_questions)}/{len(questions)} questions retained "
              f"({len(filtered_questions)/len(questions)*100:.1f}%) - Total time: {total_time:.2f}s")

        return filtered_questions

    async def evaluate(self, input_file: str, output_dir: str, field: str = ""):
        evaluate_start_time = time.time()

        print("="*60)
        print("STEP 1: Loading Questions")
        print("="*60)
        all_questions = self.load_questions(input_file)
        for q in all_questions:
            q["field"] = field

        if not all_questions:
            print("No questions to evaluate!")
            return

        print("\n" + "="*60)
        print("STEP 2: Filtering Similar Questions")
        print("="*60)
        filtered_questions = self.filter_questions(all_questions)

        os.makedirs(output_dir, exist_ok=True)
        filtered_path = os.path.join(output_dir, "filtered_questions.json")
        save_file(filtered_questions, filtered_path)
        print(f"\nFiltered questions saved to: {filtered_path}")

        print("\n" + "="*60)
        print("STEP 3: Group-Based Scoring and Ranking")
        print("="*60)
        results = await self.scorer.score_questions(filtered_questions, field)

        print("="*60)
        print("STEP 4: Saving Results")
        print("="*60)

        complete_results_path = os.path.join(output_dir, "evaluation_results.json")
        save_file(results, complete_results_path)
        print(f"Complete results saved to: {complete_results_path}")

        ranked_questions = [results['questions_with_scores'][i]
                           for i in results['consensus_ranking']]
        ranked_path = os.path.join(output_dir, "ranked_questions.json")
        save_file(ranked_questions, ranked_path)
        print(f"Ranked questions saved to: {ranked_path}")

        summary = {
            'total_input_questions': len(all_questions),
            'filtered_questions': len(filtered_questions),
            'retention_rate': len(filtered_questions) / len(all_questions),
            'models_used': self.scorer.models,
            'similarity_threshold': self.similarity_threshold,
            'comparison_rounds': self.scorer.comparison_rounds,
            'group_size': self.scorer.group_size,
            'scoring_method': 'group_based_only',
            'top_10_questions': [
                {
                    'rank': i + 1,
                    'question': ranked_questions[i]['question'],
                    'average_scores': ranked_questions[i]['average_scores']
                }
                for i in range(min(10, len(ranked_questions)))
            ]
        }
        summary_path = os.path.join(output_dir, "summary.json")
        save_file(summary, summary_path)
        print(f"Summary saved to: {summary_path}")

        evaluate_total_time = time.time() - evaluate_start_time

        print("\n" + "="*60)
        print("EVALUATION COMPLETE!")
        print("="*60)
        print(f"Input questions: {len(all_questions)}")
        print(f"After filtering: {len(filtered_questions)}")
        print(f"Models used: {', '.join(self.scorer.models)}")
        print(f"Comparison rounds: {self.scorer.comparison_rounds}")
        print(f"Group size: {self.scorer.group_size}")
        print(f"Total evaluation time: {evaluate_total_time:.2f}s ({evaluate_total_time/60:.2f} minutes)")
        print(f"\nTop 3 questions:")
        for i in range(min(3, len(ranked_questions))):
            q = ranked_questions[i]
            print(f"\n{i+1}. {q['question']}")
            print(f"   Score: {q['average_scores']['total']:.2f} "
                  f"(N:{q['average_scores']['novelty']:.1f}, "
                  f"F:{q['average_scores']['feasibility']:.1f}, "
                  f"S:{q['average_scores']['significance']:.1f})")


# ==================== Main Entry Point ====================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate scientific research questions (group-based scoring)")
    parser.add_argument("--input_file", type=str, default="./data/raw_questions/life-1.json")
    parser.add_argument("--output_dir", type=str, default="./data/evaluated_questions/life/")
    parser.add_argument("--field", type=str, default="")
    parser.add_argument("--similarity_threshold", type=float, default=0.85)
    parser.add_argument("--filter_batch_size", type=int, default=50)
    parser.add_argument("--comparison_rounds", type=int, default=3,
                        help="Number of scoring rounds (default: 3)")
    parser.add_argument("--group_size", type=int, default=5,
                        help="Size of each group for scoring (default: 5)")
    parser.add_argument("--models", type=str, nargs="+", default=["gpt-4o-mini"])
    parser.add_argument("--log_dir", type=str, default=None)
    parser.add_argument("--max_concurrent_tasks", type=int, default=32,
                        help="Maximum number of concurrent async tasks (default: 32)")

    args = parser.parse_args()

    if args.log_dir is None:
        args.log_dir = os.path.join(args.output_dir, "logs")

    evaluator = QuestionEvaluator(
        similarity_threshold=args.similarity_threshold,
        filter_batch_size=args.filter_batch_size,
        models=args.models,
        comparison_rounds=args.comparison_rounds,
        group_size=args.group_size,
        log_dir=args.log_dir,
        max_concurrent_tasks=args.max_concurrent_tasks,
    )

    await evaluator.evaluate(
        input_file=args.input_file,
        output_dir=args.output_dir,
        field=args.field
    )


if __name__ == "__main__":
    asyncio.run(main())
