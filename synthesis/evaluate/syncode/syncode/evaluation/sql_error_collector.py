"""
SQL Spider Error Data Collector

This module provides functionality to collect detailed error data from SQL predictions
on the Spider dataset. It evaluates syntax errors, execution errors, and collects
comprehensive error information for analysis.
"""

import os
import json
import sqlite3
import time
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
from tqdm import tqdm
from datasets import load_dataset

# Try relative imports first, fall back to absolute
try:
    from syncode.utils.sql_spider_eval.evaluation import (
        evaluate, eval_syntax, eval_exec_match, Evaluator, 
        build_foreign_key_map_from_json
    )
    from syncode.utils.sql_spider_eval.process_sql import get_schema, Schema, get_sql
except ImportError:
    from ..utils.sql_spider_eval.evaluation import (
        evaluate, eval_syntax, eval_exec_match, Evaluator,
        build_foreign_key_map_from_json
    )
    from ..utils.sql_spider_eval.process_sql import get_schema, Schema, get_sql


@dataclass
class SQLErrorRecord:
    """Data class to store error information for a single SQL prediction"""
    task_id: int
    db_id: str
    question: str
    predicted_sql: str
    gold_sql: str
    
    # Validity information
    validity: str  # 'Valid', 'Syntax Error', 'Other Error'
    error_message: Optional[str]
    
    # Execution information
    exec_match: bool
    hardness: str  # 'easy', 'medium', 'hard', 'extra'
    
    # Parsing information
    parse_success: bool
    parse_error: Optional[str]
    
    # Timing information
    total_time: Optional[float]
    total_tokens: Optional[int]
    
    # Additional metadata
    raw_completion: Optional[str]


class SQLErrorCollector:
    """
    Collector for SQL error data from the Spider dataset.
    
    This class provides methods to:
    - Load and process predictions against the Spider dataset
    - Collect detailed error information including syntax, execution, and parsing errors
    - Export error data in various formats (JSONL, summary statistics)
    """
    
    def __init__(self, db_dir: Optional[str] = None, tables_path: Optional[str] = None, 
                 gold_file: Optional[str] = None):
        """
        Initialize the SQL Error Collector.
        
        Args:
            db_dir: Path to the databases directory
            tables_path: Path to the tables.json file
            gold_file: Path to the gold SQL file
        """
        current_dir = os.path.dirname(os.path.realpath(__file__))
        
        self.db_dir = db_dir or f"{current_dir}/../utils/sql_spider_eval/databases"
        self.tables_path = tables_path or f"{current_dir}/../utils/sql_spider_eval/evaluation_examples/examples/tables.json"
        self.gold_file = gold_file or f"{current_dir}/../utils/sql_spider_eval/evaluation_examples/gold_example.txt"
        
        self.evaluator = Evaluator()
        self.error_records: List[SQLErrorRecord] = []
        
        # Load foreign key maps
        if os.path.exists(self.tables_path):
            self.kmaps = build_foreign_key_map_from_json(self.tables_path)
        else:
            self.kmaps = {}
            print(f"Warning: tables.json not found at {self.tables_path}")
    
    def load_spider_dataset(self) -> List[Dict]:
        """Load the Spider validation dataset from HuggingFace."""
        ds = load_dataset("richardr1126/spider-context-validation", split="validation")
        problems = []
        for problem in ds:
            prompt = f"db_id: {problem['db_id']}\ndb_info: {problem['db_info']}\nquestion: {problem['question']}\nSQL:"
            problems.append({**problem, 'prompt': prompt})
        return problems
    
    def check_syntax(self, db_path: str, sql: str) -> Tuple[str, Optional[str]]:
        """
        Check SQL syntax by attempting to execute it.
        
        Args:
            db_path: Path to the SQLite database
            sql: SQL query to check
            
        Returns:
            Tuple of (validity_status, error_message)
        """
        return eval_syntax(db_path, sql)
    
    def check_execution_match(self, db_path: str, pred_sql: str, gold_sql: str) -> bool:
        """
        Check if predicted SQL returns same results as gold SQL.
        
        Args:
            db_path: Path to the SQLite database
            pred_sql: Predicted SQL query
            gold_sql: Gold SQL query
            
        Returns:
            True if results match, False otherwise
        """
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Execute predicted SQL
            try:
                cursor.execute(pred_sql)
                pred_results = set(map(tuple, cursor.fetchall()))
            except Exception:
                return False
            
            # Execute gold SQL
            cursor.execute(gold_sql)
            gold_results = set(map(tuple, cursor.fetchall()))
            
            conn.close()
            return pred_results == gold_results
        except Exception:
            return False
    
    def get_query_hardness(self, db_path: str, gold_sql: str) -> str:
        """
        Determine the hardness level of a query.
        
        Args:
            db_path: Path to the SQLite database
            gold_sql: Gold SQL query
            
        Returns:
            Hardness level: 'easy', 'medium', 'hard', 'extra'
        """
        try:
            schema = Schema(get_schema(db_path))
            g_sql = get_sql(schema, gold_sql)
            return self.evaluator.eval_hardness(g_sql)
        except Exception:
            return 'unknown'
    
    def parse_sql(self, db_path: str, sql: str) -> Tuple[bool, Optional[str], Optional[Dict]]:
        """
        Attempt to parse SQL into structured format.
        
        Args:
            db_path: Path to the SQLite database
            sql: SQL query to parse
            
        Returns:
            Tuple of (success, error_message, parsed_sql_dict)
        """
        try:
            schema = Schema(get_schema(db_path))
            parsed = get_sql(schema, sql)
            return True, None, parsed
        except Exception as e:
            return False, str(e), None
    
    def collect_error_from_prediction(
        self,
        task_id: int,
        problem: Dict,
        predicted_sql: str,
        gold_sql: Optional[str] = None,
        raw_completion: Optional[str] = None,
        total_time: Optional[float] = None,
        total_tokens: Optional[int] = None
    ) -> SQLErrorRecord:
        """
        Collect detailed error information for a single prediction.
        
        Args:
            task_id: ID of the task
            problem: Problem dictionary containing db_id, question, etc.
            predicted_sql: The predicted SQL query
            gold_sql: The gold SQL query (if not provided, uses ground_truth from problem)
            raw_completion: The raw model completion before post-processing
            total_time: Time taken for generation
            total_tokens: Number of tokens generated
            
        Returns:
            SQLErrorRecord with all collected error information
        """
        db_id = problem['db_id']
        db_path = os.path.join(self.db_dir, db_id, f"{db_id}.sqlite")
        
        if gold_sql is None:
            gold_sql = problem.get('ground_truth', problem.get('query', ''))
        
        # Check syntax validity
        validity, error_message = self.check_syntax(db_path, predicted_sql)
        
        # Check execution match
        exec_match = False
        if validity == 'Valid':
            exec_match = self.check_execution_match(db_path, predicted_sql, gold_sql)
        
        # Get query hardness
        hardness = self.get_query_hardness(db_path, gold_sql)
        
        # Try to parse the predicted SQL
        parse_success, parse_error, _ = self.parse_sql(db_path, predicted_sql)
        
        record = SQLErrorRecord(
            task_id=task_id,
            db_id=db_id,
            question=problem.get('question', ''),
            predicted_sql=predicted_sql,
            gold_sql=gold_sql,
            validity=validity,
            error_message=error_message,
            exec_match=exec_match,
            hardness=hardness,
            parse_success=parse_success,
            parse_error=parse_error,
            total_time=total_time,
            total_tokens=total_tokens,
            raw_completion=raw_completion
        )
        
        self.error_records.append(record)
        return record
    
    def collect_errors_from_file(
        self,
        predictions_file: str,
        problems: Optional[List[Dict]] = None
    ) -> List[SQLErrorRecord]:
        """
        Collect errors from a predictions file.
        
        Args:
            predictions_file: Path to file with one SQL prediction per line
            problems: Optional list of problems (loads Spider dataset if not provided)
            
        Returns:
            List of SQLErrorRecord objects
        """
        if problems is None:
            problems = self.load_spider_dataset()
        
        # Read predictions
        with open(predictions_file, 'r') as f:
            predictions = [line.strip() for line in f.readlines() if line.strip()]
        
        # Load gold queries
        gold_queries = []
        if os.path.exists(self.gold_file):
            with open(self.gold_file, 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        parts = line.strip().split('\t')
                        gold_queries.append(parts[0] if len(parts) > 0 else '')
        
        # Collect errors
        records = []
        for i, (pred, problem) in enumerate(tqdm(zip(predictions, problems), 
                                                   total=min(len(predictions), len(problems)),
                                                   desc="Collecting errors")):
            gold = gold_queries[i] if i < len(gold_queries) else problem.get('ground_truth', '')
            record = self.collect_error_from_prediction(
                task_id=i,
                problem=problem,
                predicted_sql=pred,
                gold_sql=gold
            )
            records.append(record)
        
        return records
    
    def collect_errors_from_jsonl(
        self,
        jsonl_file: str,
        problems: Optional[List[Dict]] = None
    ) -> List[SQLErrorRecord]:
        """
        Collect errors from a JSONL file with predictions.
        
        Args:
            jsonl_file: Path to JSONL file with predictions
            problems: Optional list of problems
            
        Returns:
            List of SQLErrorRecord objects
        """
        if problems is None:
            problems = self.load_spider_dataset()
        
        # Read JSONL predictions
        predictions = []
        with open(jsonl_file, 'r') as f:
            for line in f:
                if line.strip():
                    predictions.append(json.loads(line))
        
        # Collect errors
        records = []
        for pred_data in tqdm(predictions, desc="Collecting errors"):
            task_id = pred_data.get('task_id', len(records))
            
            if task_id < len(problems):
                problem = problems[task_id]
            else:
                problem = {'db_id': 'unknown', 'question': ''}
            
            record = self.collect_error_from_prediction(
                task_id=task_id,
                problem=problem,
                predicted_sql=pred_data.get('completion', ''),
                raw_completion=pred_data.get('raw_completion', pred_data.get('completion', '')),
                total_time=pred_data.get('total_time'),
                total_tokens=pred_data.get('total_tokens')
            )
            records.append(record)
        
        return records
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """
        Compute statistics over collected error records.
        
        Returns:
            Dictionary with error statistics
        """
        if not self.error_records:
            return {"error": "No error records collected"}
        
        total = len(self.error_records)
        
        # Count by validity
        validity_counts = defaultdict(int)
        for record in self.error_records:
            validity_counts[record.validity] += 1
        
        # Count by hardness
        hardness_counts = defaultdict(int)
        exec_by_hardness = defaultdict(list)
        for record in self.error_records:
            hardness_counts[record.hardness] += 1
            exec_by_hardness[record.hardness].append(record.exec_match)
        
        # Execution accuracy by hardness
        exec_accuracy_by_hardness = {}
        for hardness, matches in exec_by_hardness.items():
            exec_accuracy_by_hardness[hardness] = sum(matches) / len(matches) if matches else 0
        
        # Parse success rate
        parse_success_count = sum(1 for r in self.error_records if r.parse_success)
        
        # Execution accuracy for valid queries
        valid_records = [r for r in self.error_records if r.validity == 'Valid']
        exec_accuracy_valid = sum(1 for r in valid_records if r.exec_match) / len(valid_records) if valid_records else 0
        
        # Overall execution accuracy
        overall_exec_accuracy = sum(1 for r in self.error_records if r.exec_match) / total
        
        # Average time and tokens (if available)
        times = [r.total_time for r in self.error_records if r.total_time is not None]
        tokens = [r.total_tokens for r in self.error_records if r.total_tokens is not None]
        
        stats = {
            "total_samples": total,
            "validity_distribution": dict(validity_counts),
            "hardness_distribution": dict(hardness_counts),
            "execution_accuracy_by_hardness": exec_accuracy_by_hardness,
            "overall_execution_accuracy": overall_exec_accuracy,
            "execution_accuracy_valid_only": exec_accuracy_valid,
            "syntax_error_rate": validity_counts.get('Syntax Error', 0) / total,
            "other_error_rate": validity_counts.get('Other Error', 0) / total,
            "valid_rate": validity_counts.get('Valid', 0) / total,
            "parse_success_rate": parse_success_count / total,
            "average_time": sum(times) / len(times) if times else None,
            "average_tokens": sum(tokens) / len(tokens) if tokens else None,
        }
        
        return stats
    
    def get_error_samples(self, error_type: str = 'Syntax Error', limit: int = 10) -> List[Dict]:
        """
        Get sample error records of a specific type.
        
        Args:
            error_type: Type of error ('Syntax Error', 'Other Error', 'Valid')
            limit: Maximum number of samples to return
            
        Returns:
            List of error record dictionaries
        """
        samples = []
        for record in self.error_records:
            if record.validity == error_type:
                samples.append(asdict(record))
                if len(samples) >= limit:
                    break
        return samples
    
    def get_failed_executions(self, limit: int = 10) -> List[Dict]:
        """
        Get samples where syntax was valid but execution didn't match.
        
        Args:
            limit: Maximum number of samples
            
        Returns:
            List of error record dictionaries
        """
        samples = []
        for record in self.error_records:
            if record.validity == 'Valid' and not record.exec_match:
                samples.append(asdict(record))
                if len(samples) >= limit:
                    break
        return samples
    
    def export_to_jsonl(self, output_path: str):
        """
        Export all error records to a JSONL file.
        
        Args:
            output_path: Path to output JSONL file
        """
        with open(output_path, 'w') as f:
            for record in self.error_records:
                f.write(json.dumps(asdict(record)) + '\n')
        print(f"Exported {len(self.error_records)} records to {output_path}")
    
    def export_summary(self, output_path: str):
        """
        Export error statistics summary to a JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        stats = self.get_error_statistics()
        
        # Add sample errors
        stats['sample_syntax_errors'] = self.get_error_samples('Syntax Error', 5)
        stats['sample_other_errors'] = self.get_error_samples('Other Error', 5)
        stats['sample_failed_executions'] = self.get_failed_executions(5)
        
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Exported summary to {output_path}")
    
    def clear_records(self):
        """Clear all collected error records."""
        self.error_records = []


def collect_sql_errors(
    predictions_file: str,
    output_path: str,
    db_dir: Optional[str] = None,
    tables_path: Optional[str] = None,
    gold_file: Optional[str] = None,
    export_summary: bool = True
) -> Dict[str, Any]:
    """
    Convenience function to collect SQL errors from a predictions file.
    
    Args:
        predictions_file: Path to predictions file (txt with one SQL per line, or JSONL)
        output_path: Path for output JSONL file
        db_dir: Path to databases directory
        tables_path: Path to tables.json
        gold_file: Path to gold SQL file
        export_summary: Whether to also export a summary JSON
        
    Returns:
        Dictionary with error statistics
    """
    collector = SQLErrorCollector(
        db_dir=db_dir,
        tables_path=tables_path,
        gold_file=gold_file
    )
    
    # Determine file type and collect errors
    if predictions_file.endswith('.jsonl'):
        collector.collect_errors_from_jsonl(predictions_file)
    else:
        collector.collect_errors_from_file(predictions_file)
    
    # Export results
    collector.export_to_jsonl(output_path)
    
    if export_summary:
        summary_path = output_path.replace('.jsonl', '_summary.json')
        collector.export_summary(summary_path)
    
    return collector.get_error_statistics()


# Integration with SQLEval class
class SQLEvalWithErrorCollection:
    """
    Extended SQL Evaluation class that includes error data collection.
    Inherits behavior from SQLEval and adds error tracking capabilities.
    """
    
    @staticmethod
    def run_eval_with_error_collection(
        syncode,
        out_path: Optional[str],
        error_out_path: Optional[str] = None,
        num_tasks: Optional[int] = None,
        debug_task_id: Optional[int] = None
    ):
        """
        Run evaluation on SQL dataset with detailed error collection.
        
        Args:
            syncode: Syncode object with model and dataset
            out_path: Path for main output
            error_out_path: Path for error data output (defaults to out_path + '_errors.jsonl')
            num_tasks: Number of tasks to run
            debug_task_id: Specific task ID to debug
        """
        from mxeval.data import write_jsonl
        
        problems = syncode.dataset.problems
        
        if num_tasks is not None:
            problems = problems[:num_tasks]
        
        samples = []
        pbar = tqdm(total=len(problems) * syncode.num_samples)
        assert syncode.num_samples == 1, "SQL evaluation only supports num_samples=1"
        
        predict_file = out_path
        error_out_path = error_out_path or (out_path.replace('.jsonl', '_errors.jsonl') if out_path else 'sql_errors.jsonl')
        
        # Initialize error collector
        collector = SQLErrorCollector()
        
        if syncode.grammar_decoder is not None:
            syncode.grammar_decoder.parse_output_only = True
        
        if debug_task_id is not None:
            problems = [problems[debug_task_id]]
        
        with open(predict_file, 'w') as f:
            for task_id, problem in enumerate(problems):
                start_time = time.time()
                batch_completions = syncode.model.generate_grammar_constrained_completion(
                    problem['prompt'],
                    syncode.num_samples
                )
                end_time = time.time()
                
                raw_completion = batch_completions[0]
                completion = syncode.dataset.post_process_answer(raw_completion)
                
                # Extract SQL from markdown if present
                extract = False
                if extract and '```' in completion:
                    completion = completion.split('```')[1]
                    if 'sql' in completion:
                        completion = completion.split('sql')[1]
                
                # Collect error data
                collector.collect_error_from_prediction(
                    task_id=task_id,
                    problem=problem,
                    predicted_sql=completion,
                    raw_completion=raw_completion,
                    total_time=end_time - start_time,
                    total_tokens=syncode.model.total_tokens
                )
                
                res = dict(
                    task_id=task_id,
                    completion=completion,
                    total_tokens=syncode.model.total_tokens,
                    total_time=end_time - start_time
                )
                samples.append(res)
                f.write(completion + '\n')
                pbar.update(syncode.num_samples)
        
        pbar.close()
        
        # Export error data
        collector.export_to_jsonl(error_out_path)
        collector.export_summary(error_out_path.replace('.jsonl', '_summary.json'))
        
        # Print statistics
        stats = collector.get_error_statistics()
        print("\n=== Error Collection Summary ===")
        print(f"Total samples: {stats['total_samples']}")
        print(f"Validity distribution: {stats['validity_distribution']}")
        print(f"Overall execution accuracy: {stats['overall_execution_accuracy']:.4f}")
        print(f"Execution accuracy by hardness: {stats['execution_accuracy_by_hardness']}")
        print(f"Syntax error rate: {stats['syntax_error_rate']:.4f}")
        print(f"Other error rate: {stats['other_error_rate']:.4f}")
        if stats['average_time']:
            print(f"Average time: {stats['average_time']:.4f}s")
        if stats['average_tokens']:
            print(f"Average tokens: {stats['average_tokens']:.2f}")
        
        if out_path is not None and debug_task_id is None:
            write_jsonl(out_path, samples)
        
        return stats, collector.error_records


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect SQL error data from predictions")
    parser.add_argument("--predictions", type=str, required=True,
                       help="Path to predictions file (txt or jsonl)")
    parser.add_argument("--output", type=str, required=True,
                       help="Path for output JSONL file")
    parser.add_argument("--db-dir", type=str, default=None,
                       help="Path to databases directory")
    parser.add_argument("--tables", type=str, default=None,
                       help="Path to tables.json")
    parser.add_argument("--gold", type=str, default=None,
                       help="Path to gold SQL file")
    parser.add_argument("--no-summary", action="store_true",
                       help="Don't export summary JSON")
    
    args = parser.parse_args()
    
    stats = collect_sql_errors(
        predictions_file=args.predictions,
        output_path=args.output,
        db_dir=args.db_dir,
        tables_path=args.tables,
        gold_file=args.gold,
        export_summary=not args.no_summary
    )
    
    print("\n=== Error Statistics ===")
    print(json.dumps(stats, indent=2))

