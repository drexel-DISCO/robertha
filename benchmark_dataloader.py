"""
Loading tasks
=================================
"""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr, spearmanr
from torchmetrics.classification import BinaryMatthewsCorrCoef
import json
from collections import defaultdict


# ============================================================================
# UNIFIED TASK CONFIGURATIONS
# ============================================================================

TASK_CONFIGS = {
    # ========== GLUE Tasks (9 tasks) ==========
    'cola': {
        'num_labels': 2,
        'metric': 'matthews_corrcoef',
        'text_keys': ['sentence'],
        'label_key': 'label',
        'dataset_name': 'glue',
        'task_name': 'cola',
        'category': 'linguistic_acceptability',
        'benchmark': 'glue'
    },
    'sst2': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['sentence'],
        'label_key': 'label',
        'dataset_name': 'glue',
        'task_name': 'sst2',
        'category': 'sentiment',
        'benchmark': 'glue'
    },
    'mrpc': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'glue',
        'task_name': 'mrpc',
        'category': 'paraphrase',
        'benchmark': 'glue'
    },
    'qqp': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['question1', 'question2'],
        'label_key': 'label',
        'dataset_name': 'glue',
        'task_name': 'qqp',
        'category': 'paraphrase',
        'benchmark': 'glue'
    },
    'stsb': {
        'num_labels': 1,
        'metric': 'pearson_spearman',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'glue',
        'task_name': 'stsb',
        'category': 'similarity',
        'benchmark': 'glue'
    },
    'mnli': {
        'num_labels': 3,
        'metric': 'accuracy',
        'text_keys': ['premise', 'hypothesis'],
        'label_key': 'label',
        'dataset_name': 'glue',
        'task_name': 'mnli',
        'category': 'nli',
        'benchmark': 'glue',
        'special_split': 'validation_matched'
    },
    'qnli': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['question', 'sentence'],
        'label_key': 'label',
        'dataset_name': 'glue',
        'task_name': 'qnli',
        'category': 'qa',
        'benchmark': 'glue'
    },
    'rte': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'glue',
        'task_name': 'rte',
        'category': 'nli',
        'benchmark': 'glue'
    },
    'wnli': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'glue',
        'task_name': 'wnli',
        'category': 'nli',
        'benchmark': 'glue'
    },
    
    # ========== AdvGLUE Tasks (Adversarial GLUE - Inference Only) ==========
    # AdvGLUE provides adversarial versions of GLUE tasks for robustness testing
    # These use the same configuration as base GLUE tasks but with adversarial examples
    # Dataset: AI-Secure/adv_glue on HuggingFace Hub
    'adv_sst2': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['sentence'],
        'label_key': 'label',
        'dataset_name': 'AI-Secure/adv_glue',
        'task_name': 'adv_sst2',
        'category': 'sentiment',
        'benchmark': 'advglue',
        'base_task': 'sst2',  # Maps to base GLUE task for model loading
        'inference_only': True
    },
    'adv_qqp': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['question1', 'question2'],
        'label_key': 'label',
        'dataset_name': 'AI-Secure/adv_glue',
        'task_name': 'adv_qqp',
        'category': 'paraphrase',
        'benchmark': 'advglue',
        'base_task': 'qqp',
        'inference_only': True
    },
    'adv_mnli': {
        'num_labels': 3,
        'metric': 'accuracy',
        'text_keys': ['premise', 'hypothesis'],
        'label_key': 'label',
        'dataset_name': 'AI-Secure/adv_glue',
        'task_name': 'adv_mnli',
        'category': 'nli',
        'benchmark': 'advglue',
        'base_task': 'mnli',
        'inference_only': True
    },
    'adv_mnli_mismatched': {
        'num_labels': 3,
        'metric': 'accuracy',
        'text_keys': ['premise', 'hypothesis'],
        'label_key': 'label',
        'dataset_name': 'AI-Secure/adv_glue',
        'task_name': 'adv_mnli_mismatched',
        'category': 'nli',
        'benchmark': 'advglue',
        'base_task': 'mnli',
        'inference_only': True
    },
    'adv_qnli': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['question', 'sentence'],
        'label_key': 'label',
        'dataset_name': 'AI-Secure/adv_glue',
        'task_name': 'adv_qnli',
        'category': 'qa',
        'benchmark': 'advglue',
        'base_task': 'qnli',
        'inference_only': True
    },
    'adv_rte': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'AI-Secure/adv_glue',
        'task_name': 'adv_rte',
        'category': 'nli',
        'benchmark': 'advglue',
        'base_task': 'rte',
        'inference_only': True
    },
    
    # ========== PAWS (Paraphrase Adversaries from Word Scrambling) ==========
    # PAWS: Adversarial paraphrase detection with high lexical overlap
    # Dataset: google-research-datasets/paws on HuggingFace Hub
    # Three configurations: labeled_final (default), labeled_swap, unlabeled_final
    'paws': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'google-research-datasets/paws',
        'task_name': 'labeled_final',  # Default configuration
        'category': 'paraphrase',
        'benchmark': 'paws'
    },
    'paws_swap': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'google-research-datasets/paws',
        'task_name': 'labeled_swap',
        'category': 'paraphrase',
        'benchmark': 'paws'
    },
    'paws_unlabeled': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'google-research-datasets/paws',
        'task_name': 'unlabeled_final',
        'category': 'paraphrase',
        'benchmark': 'paws',
        'noisy_labels': True  # This subset has noisy labels
    },
    
    # ========== PAWS-X (Multilingual PAWS) ==========
    # PAWS-X: Cross-lingual adversarial paraphrase detection
    # Dataset: google-research-datasets/paws-x on HuggingFace Hub
    # Languages: de, en, es, fr, ja, ko, zh
    'pawsx_en': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'google-research-datasets/paws-x',
        'task_name': 'en',
        'category': 'paraphrase',
        'benchmark': 'pawsx',
        'language': 'en'
    },
    'pawsx_de': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'google-research-datasets/paws-x',
        'task_name': 'de',
        'category': 'paraphrase',
        'benchmark': 'pawsx',
        'language': 'de'
    },
    'pawsx_es': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'google-research-datasets/paws-x',
        'task_name': 'es',
        'category': 'paraphrase',
        'benchmark': 'pawsx',
        'language': 'es'
    },
    'pawsx_fr': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'google-research-datasets/paws-x',
        'task_name': 'fr',
        'category': 'paraphrase',
        'benchmark': 'pawsx',
        'language': 'fr'
    },
    'pawsx_zh': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'google-research-datasets/paws-x',
        'task_name': 'zh',
        'category': 'paraphrase',
        'benchmark': 'pawsx',
        'language': 'zh'
    },
    'pawsx_ja': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'google-research-datasets/paws-x',
        'task_name': 'ja',
        'category': 'paraphrase',
        'benchmark': 'pawsx',
        'language': 'ja'
    },
    'pawsx_ko': {
        'num_labels': 2,
        'metric': 'f1',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'google-research-datasets/paws-x',
        'task_name': 'ko',
        'category': 'paraphrase',
        'benchmark': 'pawsx',
        'language': 'ko'
    },
    
    # ========== SuperGLUE Tasks (8 tasks) ==========
    'boolq': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['passage', 'question'],
        'label_key': 'label',
        'dataset_name': 'super_glue',
        'task_name': 'boolq',
        'category': 'reading_comprehension',
        'benchmark': 'superglue'
    },
    'cb': {
        'num_labels': 3,
        'metric': 'accuracy_and_f1',  # Official: both accuracy and macro-F1
        'text_keys': ['premise', 'hypothesis'],
        'label_key': 'label',
        'dataset_name': 'super_glue',
        'task_name': 'cb',
        'category': 'nli',
        'benchmark': 'superglue'
    },
    'copa': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['premise', 'choice1', 'choice2', 'question'],
        'label_key': 'label',
        'dataset_name': 'super_glue',
        'task_name': 'copa',
        'special_format': True,
        'category': 'reasoning',
        'benchmark': 'superglue',
        'requires_choice_id': True  # Track which question examples belong to
    },
    'multirc': {
        'num_labels': 2,
        'metric': 'f1a',  # FIXED: Use F1a (per-question F1 average), not regular F1
        'text_keys': ['paragraph', 'question', 'answer'],
        'label_key': 'label',
        'dataset_name': 'super_glue',
        'task_name': 'multirc',
        'special_format': True,
        'category': 'reading_comprehension',
        'benchmark': 'superglue',
        'requires_question_id': True  # NEW: Flag indicating need for question grouping
    },
    'record': {
        'num_labels': 2,
        'metric': 'f1',  # Binary F1 on entity candidates (simplified)
        'text_keys': ['passage', 'query', 'entities'],
        'label_key': 'answers',
        'dataset_name': 'super_glue',
        'task_name': 'record',
        'special_format': True,
        'category': 'reading_comprehension',
        'benchmark': 'superglue',
        'note': 'Using entity ranking with binary F1 (simplified from official token F1+EM)'
    },
    'wic': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['sentence1', 'sentence2', 'word'],
        'label_key': 'label',
        'dataset_name': 'super_glue',
        'task_name': 'wic',
        'category': 'word_sense',
        'benchmark': 'superglue'
    },
    'wsc': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['text', 'span1_text', 'span2_text'],
        'label_key': 'label',
        'dataset_name': 'super_glue',
        'task_name': 'wsc',
        'special_format': True,
        'category': 'reasoning',
        'benchmark': 'superglue'
    },
    'axb': {
        'num_labels': 2,
        'metric': 'matthews_corrcoef',
        'text_keys': ['sentence1', 'sentence2'],
        'label_key': 'label',
        'dataset_name': 'super_glue',
        'task_name': 'axb',
        'category': 'diagnostic',
        'benchmark': 'superglue',
        'test_only': True
    },
    'axg': {
        'num_labels': 2,
        'metric': 'accuracy',
        'text_keys': ['premise', 'hypothesis'],
        'label_key': 'label',
        'dataset_name': 'super_glue',
        'task_name': 'axg',
        'category': 'diagnostic',
        'benchmark': 'superglue',
        'test_only': True
    },
    
    # ========== SQuAD v2 (Reading Comprehension) ==========
    'squad_v2': {
        'num_labels': 2,  # QA task: start and end positions (set to 2 for model compatibility)
        'metric': 'squad_f1_em',  # FIXED: Use F1 and EM metrics
        'text_keys': ['question', 'context'],
        'label_key': 'answers',
        'dataset_name': 'squad_v2',
        'task_name': None,
        'special_format': True,
        'category': 'reading_comprehension',
        'benchmark': 'squad',
        'requires_answer_spans': True,  # NEW: Flag for span-based QA
        'is_qa_task': True,  # Flag indicating this is a QA task, not classification
        'requires_qa_model': True  # Requires model with start/end position heads
    },
    
    # ========== HotpotQA (Multi-hop Question Answering) ==========
    # HotpotQA: Multi-hop reasoning QA with adversarial distractors
    # Dataset: hotpotqa/hotpot_qa on HuggingFace Hub
    # Two configurations: distractor (with adversarial distractors), fullwiki (full retrieval)
    'hotpotqa': {
        'num_labels': 2,  # QA task: extractive QA (set to 2 for model compatibility)
        'metric': 'squad_f1_em',  # Use F1 and EM metrics like SQuAD
        'text_keys': ['question', 'context'],
        'label_key': 'answer',
        'dataset_name': 'hotpotqa/hotpot_qa',
        'task_name': 'distractor',  # Use distractor configuration (with adversarial context)
        'special_format': True,
        'category': 'reading_comprehension',
        'benchmark': 'hotpotqa',
        'requires_answer_spans': True,
        'is_qa_task': True,
        'requires_qa_model': True,
        'multi_hop': True  # Requires multi-hop reasoning
    },
    'hotpotqa_fullwiki': {
        'num_labels': 2,
        'metric': 'squad_f1_em',
        'text_keys': ['question', 'context'],
        'label_key': 'answer',
        'dataset_name': 'hotpotqa/hotpot_qa',
        'task_name': 'fullwiki',  # Full Wikipedia retrieval setting
        'special_format': True,
        'category': 'reading_comprehension',
        'benchmark': 'hotpotqa',
        'requires_answer_spans': True,
        'is_qa_task': True,
        'requires_qa_model': True,
        'multi_hop': True
    },
    
    # ========== ANLI (Adversarial Natural Language Inference) ==========
    # ANLI: Adversarially-constructed NLI dataset with 3 rounds of increasing difficulty
    # Dataset: anli (facebook/anli) on HuggingFace Hub
    # Each round was created with increasingly sophisticated adversarial methods:
    #   R1: Adversarial examples created to fool BERT-based models
    #   R2: Adversarial examples created to fool RoBERTa-based models  
    #   R3: Most difficult - adversarial examples + human verification
    # 3-way classification: entailment (0), neutral (1), contradiction (2)
    'anli_r1': {
        'num_labels': 3,
        'metric': 'accuracy',
        'text_keys': ['premise', 'hypothesis'],
        'label_key': 'label',
        'dataset_name': 'anli',
        'task_name': None,  # Uses split names directly
        'category': 'nli',
        'benchmark': 'anli',
        'round': 1,
        'special_splits': True  # Uses train_r1, dev_r1, test_r1
    },
    'anli_r2': {
        'num_labels': 3,
        'metric': 'accuracy',
        'text_keys': ['premise', 'hypothesis'],
        'label_key': 'label',
        'dataset_name': 'anli',
        'task_name': None,
        'category': 'nli',
        'benchmark': 'anli',
        'round': 2,
        'special_splits': True  # Uses train_r2, dev_r2, test_r2
    },
    'anli_r3': {
        'num_labels': 3,
        'metric': 'accuracy',
        'text_keys': ['premise', 'hypothesis'],
        'label_key': 'label',
        'dataset_name': 'anli',
        'task_name': None,
        'category': 'nli',
        'benchmark': 'anli',
        'round': 3,
        'special_splits': True  # Uses train_r3, dev_r3, test_r3
    },
    
    # ========== SWAG (Situations With Adversarial Generations) ==========
    # SWAG: Grounded commonsense inference - predict what happens next
    # Dataset: swag or allenai/swag on HuggingFace Hub
    # Multiple choice: given a partial description, choose the most likely continuation
    'swag': {
        'num_labels': 2,  # Binary classification per choice (like COPA)
        'metric': 'accuracy',
        'text_keys': ['sent1', 'sent2', 'ending0', 'ending1', 'ending2', 'ending3'],
        'label_key': 'label',
        'dataset_name': 'swag',
        'task_name': 'regular',  # Use regular configuration
        'special_format': True,
        'category': 'commonsense_reasoning',
        'benchmark': 'swag',
        'requires_choice_id': True  # Track which question examples belong to
    },
}

# Add the rest of tasks from original file (hellaswag, winogrande, etc.) here if needed
# For brevity, only showing the critical GLUE/SuperGLUE/SQuAD tasks


# ============================================================================
# ENHANCED DATASET CLASS WITH METADATA SUPPORT
# ============================================================================

class BenchmarkDataset(Dataset):
    """
    Enhanced PyTorch Dataset that supports metadata (e.g., question IDs for MultiRC).
    """
    
    def __init__(self, encodings, labels, metadata=None):
        """
        Args:
            encodings: Tokenized inputs
            labels: Target labels
            metadata: Optional dict with additional info (e.g., {'question_id': [...]})
        """
        self.encodings = encodings
        self.labels = labels
        self.metadata = metadata or {}
    
    def __getitem__(self, idx):
        item = {}
        for key, val in self.encodings.items():
            if key != 'label':
                item[key] = torch.tensor(val[idx])
        item['label'] = torch.tensor(self.labels[idx])
        
        # Add metadata if available
        for key, val in self.metadata.items():
            item[key] = val[idx]
        
        return item
    
    def __len__(self):
        return len(self.labels)


# ============================================================================
# TEXT PREPROCESSING FUNCTIONS
# ============================================================================

def format_text_pair(text1: str, text2: str = None, separator: str = " [SEP] ") -> str:
    """Format single text or text pair for encoding."""
    if text2 is None:
        return str(text1)
    return f"{text1}{separator}{text2}"


def preprocess_multirc(example: Dict) -> Tuple[str, int, str]:
    """
    FIXED: Now returns (text, label, question_id) for proper F1a computation.
    
    Returns:
        text: Formatted input text
        label: Binary label
        question_id: Unique identifier for the question (paragraph + question text)
    """
    text = format_text_pair(
        f"{example['paragraph']} Question: {example['question']}",
        f"Answer: {example['answer']}"
    )
    label = int(example['label'])
    
    # Create unique question ID from paragraph and question
    # Use idx if available, otherwise hash of paragraph+question
    if 'idx' in example and 'question' in example['idx']:
        question_id = f"{example['idx']['paragraph']}_{example['idx']['question']}"
    else:
        # Fallback: create ID from text hash
        question_id = str(hash(f"{example['paragraph']}|||{example['question']}"))
    
    return text, label, question_id


def preprocess_squad_v2(example: Dict) -> Tuple[str, Dict]:
    """
    FIXED: Proper SQuAD v2 preprocessing for extractive QA.
    
    Returns:
        text: Formatted question + context
        answer_info: Dict with 'text' (answer string) and 'start' (char position)
                    Empty dict if unanswerable
    """
    text = format_text_pair(example['question'], example['context'])
    
    # Extract answer information
    answers = example['answers']
    if answers['text']:  # Answerable question
        answer_info = {
            'text': answers['text'][0],  # First answer
            'start': answers['answer_start'][0]
        }
    else:  # Unanswerable question
        answer_info = {
            'text': '',
            'start': -1
        }
    
    return text, answer_info


def preprocess_hotpotqa(example: Dict) -> Tuple[str, Dict]:
    """
    Preprocess HotpotQA examples for multi-hop question answering.
    
    HotpotQA requires reasoning over multiple documents to answer questions.
    The context contains multiple paragraphs with titles and sentences.
    
    Returns:
        text: Formatted question + concatenated context
        answer_info: Dict with 'text' (answer string)
    """
    question = example['question']
    answer = example['answer']
    
    # Extract and concatenate context
    # Context format: {'title': [...], 'sentences': [[...], [...]]}
    context_data = example['context']
    titles = context_data['title']
    sentences_list = context_data['sentences']
    
    # Build context by concatenating all paragraphs
    context_parts = []
    for title, sentences in zip(titles, sentences_list):
        # Join sentences for this paragraph
        paragraph = ' '.join(sentences)
        # Add with title
        context_parts.append(f"{title}: {paragraph}")
    
    # Concatenate all paragraphs
    context = ' '.join(context_parts)
    
    # Format as question + context pair
    text = format_text_pair(question, context)
    
    # Answer information
    answer_info = {
        'text': answer
    }
    
    return text, answer_info


def preprocess_record(example: Dict) -> List[Tuple[str, int]]:
    """
    Preprocess ReCoRD examples.
    ReCoRD: Reading comprehension where you predict which entity fills @placeholder.
    
    **Returns multiple examples** - one per candidate entity.
    
    Returns:
        List of (text, label) tuples - one per candidate entity
        text: passage + query with entity filled in
        label: 1 if entity is correct answer, 0 otherwise
    """
    passage = example['passage']
    query = example['query']
    entities = example['entities']
    answers = example['answers']
    
    examples = []
    
    # Create one training example per candidate entity
    for entity in entities:
        # Replace @placeholder with candidate entity
        filled_query = query.replace('@placeholder', entity)
        text = format_text_pair(passage, filled_query)
        
        # Label is 1 if this entity is in the answer list
        label = 1 if entity in answers else 0
        examples.append((text, label))
    
    return examples


def preprocess_copa(example: Dict) -> List[Tuple[str, int, str, int]]:
    """
    Preprocess COPA examples.
    COPA: Given a premise and question (cause/effect), choose between two alternatives.
    
    **Standard approach: Score each choice separately.**
    Returns two examples per input - one for each choice.
    
    Returns:
        List of (text, label, copa_id, choice_num) tuples
        text: premise + question + single choice
        label: 1 if this is correct choice, 0 otherwise
        copa_id: unique ID for this COPA question
        choice_num: 0 or 1 (which choice this is)
    """
    premise = example['premise']
    choice1 = example['choice1']
    choice2 = example['choice2']
    question = example['question']  # "cause" or "effect"
    correct_choice = int(example['label'])  # 0 = choice1, 1 = choice2
    
    # Create unique ID for this COPA question
    copa_id = str(example.get('idx', hash(premise)))
    
    examples = []
    
    # Example 1: premise + question + choice1
    text1 = f"{premise} [SEP] What was the {question}? [SEP] {choice1}"
    label1 = 1 if correct_choice == 0 else 0
    examples.append((text1, label1, copa_id, 0))
    
    # Example 2: premise + question + choice2
    text2 = f"{premise} [SEP] What was the {question}? [SEP] {choice2}"
    label2 = 1 if correct_choice == 1 else 0
    examples.append((text2, label2, copa_id, 1))
    
    return examples


def preprocess_swag(example: Dict) -> List[Tuple[str, int, str, int]]:
    """
    Preprocess SWAG examples.
    SWAG: Given a partial description, choose the most likely continuation.
    
    **Returns 4 examples per input - one for each ending choice.**
    
    Returns:
        List of (text, label, swag_id, choice_num) tuples
        text: context + ending
        label: 1 if this is correct ending, 0 otherwise
        swag_id: unique ID for this SWAG question
        choice_num: 0, 1, 2, or 3 (which ending this is)
    """
    # SWAG has sent1 (always present) and sent2 (may be empty)
    sent1 = example['sent1']
    sent2 = example.get('sent2', '')
    
    # Build context from sent1 and sent2
    if sent2:
        context = f"{sent1} {sent2}"
    else:
        context = sent1
    
    # Four ending options
    endings = [
        example['ending0'],
        example['ending1'],
        example['ending2'],
        example['ending3']
    ]
    
    correct_choice = int(example['label'])  # 0, 1, 2, or 3
    
    # Create unique ID for this SWAG question
    swag_id = str(example.get('video-id', '') + '_' + example.get('fold-ind', ''))
    if not swag_id.strip('_'):
        swag_id = str(hash(context))
    
    examples = []
    
    # Create one example for each ending
    for choice_num, ending in enumerate(endings):
        # Format: context [SEP] ending
        text = f"{context} [SEP] {ending}"
        label = 1 if choice_num == correct_choice else 0
        examples.append((text, label, swag_id, choice_num))
    
    return examples


def preprocess_wsc(example: Dict) -> Tuple[str, int]:
    """
    Preprocess WSC examples.
    WSC: Determine if a pronoun refers to a specific noun phrase.
    
    Returns:
        text: Formatted text with marked spans
        label: Binary (0 or 1)
    """
    text_str = example['text']
    span1 = example['span1_text']
    span2 = example['span2_text']
    
    # Format: text [SEP] Does 'span2' refer to 'span1'?
    text = f"{text_str} [SEP] Does '{span2}' refer to '{span1}'?"
    label = int(example['label'])
    
    return text, label


def preprocess_wic(example: Dict) -> Tuple[str, int]:
    """
    Preprocess WiC (Word-in-Context) examples.
    WiC: Determine if a word has the same meaning in two sentences.
    
    Returns:
        text: Formatted sentences with target word
        label: Binary (0 or 1)
    """
    sentence1 = example['sentence1']
    sentence2 = example['sentence2']
    word = example['word']
    
    # Format: sentence1 [SEP] sentence2 [SEP] Word: word
    text = f"{sentence1} [SEP] {sentence2} [SEP] Word: {word}"
    label = int(example['label'])
    
    return text, label


# Simplified preprocessing functions for other tasks
def preprocess_glue_single(example: Dict, text_key: str) -> Tuple[str, int]:
    text = example[text_key]
    label = int(example['label'])
    return text, label

def preprocess_glue_pair(example: Dict, text_key1: str, text_key2: str) -> Tuple[str, int]:
    text = format_text_pair(example[text_key1], example[text_key2])
    label = int(example['label'])
    return text, label

# Create mapping dictionary (add all other tasks as needed)
PREPROCESS_FUNCTIONS = {
    'cola': lambda x: preprocess_glue_single(x, 'sentence'),
    'sst2': lambda x: preprocess_glue_single(x, 'sentence'),
    'mrpc': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'qqp': lambda x: preprocess_glue_pair(x, 'question1', 'question2'),
    'stsb': lambda x: (format_text_pair(x['sentence1'], x['sentence2']), float(x['label'])),
    'mnli': lambda x: preprocess_glue_pair(x, 'premise', 'hypothesis'),
    'qnli': lambda x: preprocess_glue_pair(x, 'question', 'sentence'),
    'rte': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'wnli': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'boolq': lambda x: preprocess_glue_pair(x, 'passage', 'question'),
    'cb': lambda x: preprocess_glue_pair(x, 'premise', 'hypothesis'),
    'copa': preprocess_copa,
    'swag': preprocess_swag,  # SWAG: 4-way multiple choice commonsense reasoning
    'wic': preprocess_wic,
    'wsc': preprocess_wsc,
    'multirc': preprocess_multirc,  # Uses special version with question_id
    'record': preprocess_record,  # ReCoRD reading comprehension
    'squad_v2': preprocess_squad_v2,  # FIXED: Uses special version with answer spans
    'hotpotqa': preprocess_hotpotqa,  # Multi-hop QA with adversarial distractors
    'hotpotqa_fullwiki': preprocess_hotpotqa,  # Same preprocessing for fullwiki config
    # ANLI tasks (use same preprocessing as MNLI - premise/hypothesis pairs)
    'anli_r1': lambda x: preprocess_glue_pair(x, 'premise', 'hypothesis'),
    'anli_r2': lambda x: preprocess_glue_pair(x, 'premise', 'hypothesis'),
    'anli_r3': lambda x: preprocess_glue_pair(x, 'premise', 'hypothesis'),
    'axb': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'axg': lambda x: preprocess_glue_pair(x, 'premise', 'hypothesis'),
    # AdvGLUE tasks (use same preprocessing as base GLUE tasks)
    'adv_sst2': lambda x: preprocess_glue_single(x, 'sentence'),
    'adv_qqp': lambda x: preprocess_glue_pair(x, 'question1', 'question2'),
    'adv_mnli': lambda x: preprocess_glue_pair(x, 'premise', 'hypothesis'),
    'adv_mnli_mismatched': lambda x: preprocess_glue_pair(x, 'premise', 'hypothesis'),
    'adv_qnli': lambda x: preprocess_glue_pair(x, 'question', 'sentence'),
    'adv_rte': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    # PAWS tasks (use same preprocessing as MRPC - sentence pair classification)
    'paws': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'paws_swap': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'paws_unlabeled': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    # PAWS-X tasks (multilingual - same preprocessing)
    'pawsx_en': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'pawsx_de': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'pawsx_es': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'pawsx_fr': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'pawsx_zh': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'pawsx_ja': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    'pawsx_ko': lambda x: preprocess_glue_pair(x, 'sentence1', 'sentence2'),
    # Add other tasks as needed
}


# ============================================================================
# MAIN DATA LOADING FUNCTION
# ============================================================================

def load_data(task_name: str, tokenizer, max_length: int = 128, max_train_samples: Optional[int] = None) -> Tuple[BenchmarkDataset, BenchmarkDataset]:
    """
    Load and preprocess any benchmark dataset.
    FIXED: Now properly handles MultiRC metadata for F1a computation and test-only tasks.
    """
    
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}. Available: {list(TASK_CONFIGS.keys())}")
    
    task_config = TASK_CONFIGS[task_name]
    requires_question_id = task_config.get('requires_question_id', False)
    requires_choice_id = task_config.get('requires_choice_id', False)
    is_test_only = task_config.get('test_only', False)
    is_inference_only = task_config.get('inference_only', False)
    
    print(f"Loading task: {task_name} ({task_config['benchmark'].upper()})")
    
    # Load dataset
    dataset_name = task_config['dataset_name']
    task_name_param = task_config.get('task_name')
    
    if task_name_param:
        dataset = load_dataset(dataset_name, task_name_param)
    else:
        dataset = load_dataset(dataset_name)
    
    # Get preprocessing function
    preprocess_fn = PREPROCESS_FUNCTIONS[task_name]
    
    # Handle test-only datasets (axb, axg)
    if is_test_only:
        print("Processing test data (diagnostic task)...")
        test_texts = []
        test_labels = []
        test_question_ids = [] if requires_question_id else None
        test_copa_ids = [] if requires_choice_id else None
        test_choice_nums = [] if requires_choice_id else None
        
        for example in dataset['test']:
            try:
                result = preprocess_fn(example)
                if task_name == 'record':
                    for text, label in result:
                        test_texts.append(text)
                        test_labels.append(label)
                elif task_name == 'copa':
                    for text, label, copa_id, choice_num in result:
                        test_texts.append(text)
                        test_labels.append(label)
                        test_copa_ids.append(copa_id)
                        test_choice_nums.append(choice_num)
                elif task_name == 'swag':
                    for text, label, swag_id, choice_num in result:
                        test_texts.append(text)
                        test_labels.append(label)
                        test_copa_ids.append(swag_id)  # Reuse copa_id list for swag_id
                        test_choice_nums.append(choice_num)
                elif requires_question_id:
                    text, label, question_id = result
                    test_question_ids.append(question_id)
                    test_texts.append(text)
                    test_labels.append(label)
                else:
                    text, label = result
                    test_texts.append(text)
                    test_labels.append(label)
            except Exception as e:
                print(f"Warning: Skipping example: {e}")
                continue
        
        # Tokenize
        test_encodings = tokenizer(
            test_texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors=None
        )
        
        # Create test dataset with metadata
        test_metadata = {}
        if requires_question_id:
            test_metadata['question_id'] = test_question_ids
        if requires_choice_id:
            test_metadata['copa_id'] = test_copa_ids
            test_metadata['choice_num'] = test_choice_nums
        
        test_dataset = BenchmarkDataset(test_encodings, test_labels, test_metadata if test_metadata else None)
        
        print(f"✓ Loaded {len(test_dataset)} test examples (no training data)")
        
        # Return None for train_dataset to signal test-only
        return None, test_dataset
    
    # Handle inference-only datasets (AdvGLUE)
    if is_inference_only:
        print("Processing inference-only dataset (no training data)...")
        val_texts = []
        val_labels = []
        val_question_ids = [] if requires_question_id else None
        val_copa_ids = [] if requires_choice_id else None
        val_choice_nums = [] if requires_choice_id else None
        
        # AdvGLUE datasets typically use 'validation' or 'dev' split
        val_split = 'validation' if 'validation' in dataset else 'dev'
        
        for example in dataset[val_split]:
            try:
                result = preprocess_fn(example)
                if task_name == 'record':
                    for text, label in result:
                        val_texts.append(text)
                        val_labels.append(label)
                elif task_name == 'copa':
                    for text, label, copa_id, choice_num in result:
                        val_texts.append(text)
                        val_labels.append(label)
                        val_copa_ids.append(copa_id)
                        val_choice_nums.append(choice_num)
                elif task_name == 'swag':
                    for text, label, swag_id, choice_num in result:
                        val_texts.append(text)
                        val_labels.append(label)
                        val_copa_ids.append(swag_id)  # Reuse copa_id list for swag_id
                        val_choice_nums.append(choice_num)
                elif requires_question_id:
                    text, label, question_id = result
                    val_question_ids.append(question_id)
                    val_texts.append(text)
                    val_labels.append(label)
                else:
                    text, label = result
                    val_texts.append(text)
                    val_labels.append(label)
            except Exception as e:
                print(f"Warning: Skipping example: {e}")
                continue
        
        # Tokenize
        val_encodings = tokenizer(
            val_texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors=None
        )
        
        # Create validation dataset with metadata
        val_metadata = {}
        if requires_question_id:
            val_metadata['question_id'] = val_question_ids
        if requires_choice_id:
            val_metadata['copa_id'] = val_copa_ids
            val_metadata['choice_num'] = val_choice_nums
        
        val_dataset = BenchmarkDataset(val_encodings, val_labels, val_metadata if val_metadata else None)
        
        print(f"✓ Loaded {len(val_dataset)} validation examples (inference-only, no training data)")
        
        # Return None for train_dataset to signal inference-only
        return None, val_dataset
    
    # Get splits
    # Special handling for ANLI which uses round-specific splits
    if task_name.startswith('anli_'):
        round_num = task_config.get('round')
        train_split = f'train_r{round_num}'
        val_split = f'dev_r{round_num}'
        print(f"  Using ANLI Round {round_num} splits: {train_split}, {val_split}")
    else:
        val_split = task_config.get('special_split', 'validation')
        train_split = 'train'
    
    # Process training data
    print("Processing training data...")
    train_texts = []
    train_labels = []
    train_question_ids = [] if requires_question_id else None
    train_copa_ids = [] if requires_choice_id else None
    train_choice_nums = [] if requires_choice_id else None
    
    train_data = dataset[train_split]
    if max_train_samples is not None and max_train_samples < len(train_data):
        print(f"  Using subset: {max_train_samples} of {len(train_data)} samples")
        train_data = train_data.select(range(max_train_samples))
    
    for example in train_data:
        try:
            result = preprocess_fn(example)
            
            # Handle ReCoRD which returns multiple examples per input
            if task_name == 'record':
                for text, label in result:
                    train_texts.append(text)
                    train_labels.append(label)
            # Handle COPA which returns two examples per input (one per choice)
            elif task_name == 'copa':
                for text, label, copa_id, choice_num in result:
                    train_texts.append(text)
                    train_labels.append(label)
                    train_copa_ids.append(copa_id)
                    train_choice_nums.append(choice_num)
            # Handle SWAG which returns four examples per input (one per choice)
            elif task_name == 'swag':
                for text, label, swag_id, choice_num in result:
                    train_texts.append(text)
                    train_labels.append(label)
                    train_copa_ids.append(swag_id)  # Reuse copa_id list for swag_id
                    train_choice_nums.append(choice_num)
            elif requires_question_id:
                text, label, question_id = result
                train_question_ids.append(question_id)
                train_texts.append(text)
                train_labels.append(label)
            else:
                text, label = result
                train_texts.append(text)
                train_labels.append(label)
        except Exception as e:
            print(f"Warning: Skipping training example: {e}")
            continue
    
    # Process validation data
    print("Processing validation data...")
    val_texts = []
    val_labels = []
    val_question_ids = [] if requires_question_id else None
    val_copa_ids = [] if requires_choice_id else None
    val_choice_nums = [] if requires_choice_id else None
    
    for example in dataset[val_split]:
        try:
            result = preprocess_fn(example)
            
            # Handle ReCoRD which returns multiple examples per input
            if task_name == 'record':
                for text, label in result:
                    val_texts.append(text)
                    val_labels.append(label)
            # Handle COPA which returns two examples per input (one per choice)
            elif task_name == 'copa':
                for text, label, copa_id, choice_num in result:
                    val_texts.append(text)
                    val_labels.append(label)
                    val_copa_ids.append(copa_id)
                    val_choice_nums.append(choice_num)
            # Handle SWAG which returns four examples per input (one per choice)
            elif task_name == 'swag':
                for text, label, swag_id, choice_num in result:
                    val_texts.append(text)
                    val_labels.append(label)
                    val_copa_ids.append(swag_id)  # Reuse copa_id list for swag_id
                    val_choice_nums.append(choice_num)
            elif requires_question_id:
                text, label, question_id = result
                val_question_ids.append(question_id)
                val_texts.append(text)
                val_labels.append(label)
            else:
                text, label = result
                val_texts.append(text)
                val_labels.append(label)
        except Exception as e:
            print(f"Warning: Skipping validation example: {e}")
            continue
    
    # Tokenize
    print("Tokenizing...")
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors=None
    )
    
    val_encodings = tokenizer(
        val_texts,
        truncation=True,
        padding='max_length',
        max_length=max_length,
        return_tensors=None
    )
    
    # Create datasets with metadata
    train_metadata = {}
    val_metadata = {}
    
    if requires_question_id:
        train_metadata['question_id'] = train_question_ids
        val_metadata['question_id'] = val_question_ids
    
    if requires_choice_id:
        train_metadata['copa_id'] = train_copa_ids
        train_metadata['choice_num'] = train_choice_nums
        val_metadata['copa_id'] = val_copa_ids
        val_metadata['choice_num'] = val_choice_nums
    
    train_dataset = BenchmarkDataset(train_encodings, train_labels, train_metadata if train_metadata else None)
    val_dataset = BenchmarkDataset(val_encodings, val_labels, val_metadata if val_metadata else None)
    
    print(f"✓ Loaded {len(train_dataset)} training examples")
    print(f"✓ Loaded {len(val_dataset)} validation examples")
    if requires_question_id:
        num_unique_questions = len(set(val_question_ids))
        print(f"✓ Unique questions in validation: {num_unique_questions}")
    if requires_choice_id:
        num_unique_copa = len(set(val_copa_ids))
        print(f"✓ Unique COPA questions in validation: {num_unique_copa}")
        print(f"  (Each question has 2 choice examples)")
    
    return train_dataset, val_dataset


# ============================================================================
# METRICS COMPUTATION WITH F1a SUPPORT
# ============================================================================

def compute_f1a_multirc(predictions, labels, question_ids):
    """
    Compute F1a for MultiRC: average of per-question F1 scores.
    
    Args:
        predictions: List of predictions
        labels: List of labels
        question_ids: List of question identifiers
    
    Returns:
        F1a score (float)
    """
    # Group by question
    question_groups = defaultdict(lambda: {'preds': [], 'labels': []})
    for pred, label, qid in zip(predictions, labels, question_ids):
        question_groups[qid]['preds'].append(pred)
        question_groups[qid]['labels'].append(label)
    
    # Compute F1 for each question
    f1_scores = []
    for qid, data in question_groups.items():
        # Compute binary F1 for this question's answers
        f1 = f1_score(data['labels'], data['preds'], average='binary', zero_division=0)
        f1_scores.append(f1)
    
    # Return average F1 across questions
    return sum(f1_scores) / len(f1_scores) if f1_scores else 0.0


def normalize_answer(s):
    """
    Normalize answer text for SQuAD evaluation.
    Lowercase and remove punctuation, articles and extra whitespace.
    """
    import re
    import string
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_squad_f1(prediction, ground_truth):
    """
    Compute token-level F1 score between prediction and ground truth.
    """
    pred_tokens = normalize_answer(prediction).split()
    truth_tokens = normalize_answer(ground_truth).split()
    
    # Handle empty predictions/ground truth
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return int(pred_tokens == truth_tokens)
    
    common_tokens = set(pred_tokens) & set(truth_tokens)
    
    if len(common_tokens) == 0:
        return 0.0
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(truth_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def compute_squad_em(prediction, ground_truth):
    """
    Compute exact match score between prediction and ground truth.
    """
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_squad_metrics(predictions, ground_truths):
    """
    Compute F1 and EM metrics for SQuAD.
    
    Args:
        predictions: List of predicted answer strings
        ground_truths: List of ground truth answer strings (or lists of valid answers)
    
    Returns:
        Dict with 'f1' and 'em' scores
    """
    f1_scores = []
    em_scores = []
    
    for pred, truth in zip(predictions, ground_truths):
        # Handle multiple valid answers (take max F1/EM)
        if isinstance(truth, list):
            f1 = max(compute_squad_f1(pred, t) for t in truth)
            em = max(compute_squad_em(pred, t) for t in truth)
        else:
            f1 = compute_squad_f1(pred, truth)
            em = compute_squad_em(pred, truth)
        
        f1_scores.append(f1)
        em_scores.append(em)
    
    return {
        'f1': sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        'em': sum(em_scores) / len(em_scores) if em_scores else 0.0
    }


def compute_metrics(task_name: str, predictions, labels, metadata=None, include_accuracy: bool = False) -> Dict[str, float]:
    """
    Compute metrics for any benchmark task.
    FIXED: Now properly computes F1a for MultiRC, F1/EM for SQuAD v2, and accuracy+F1 for CB.
    
    Args:
        task_name: Name of the task
        predictions: Model predictions 
                    - For classification: class labels (0, 1, etc.)
                    - For COPA: Either class labels OR probabilities for class 1 (preferred)
                    - For QA: predicted answer strings
        labels: Ground truth labels (class labels for classification, answer dicts for QA)
        metadata: Optional metadata (e.g., question IDs for MultiRC, copa_id for COPA)
        include_accuracy: If True, include accuracy alongside task-specific metric
    
    Returns:
        Dictionary of metrics
        
    Note for COPA:
        - If predictions are class labels (0/1), the function will pick the choice predicted as 1
        - For better results, pass probabilities/logits for class 1 instead
        - This allows proper comparison of which choice the model prefers
    """
    
    task_config = TASK_CONFIGS[task_name]
    metric_type = task_config['metric']
    
    if metric_type == 'accuracy':
        # Special handling for COPA - need to score choices (2-way)
        if task_name == 'copa' and metadata and 'copa_id' in metadata:
            # Group predictions by copa_id
            copa_groups = defaultdict(lambda: {'preds': [], 'labels': [], 'choice_nums': []})
            for pred, label, copa_id, choice_num in zip(predictions, labels, 
                                                         metadata['copa_id'], metadata['choice_num']):
                copa_groups[copa_id]['preds'].append(pred)
                copa_groups[copa_id]['labels'].append(label)
                copa_groups[copa_id]['choice_nums'].append(choice_num)
            
            # For each COPA question, pick choice with higher score/prediction
            correct = 0
            total = 0
            for copa_id, data in copa_groups.items():
                # If preds are probabilities/logits, pick higher one
                # If preds are binary labels, pick the one predicted as 1
                if data['preds'][0] > data['preds'][1]:
                    predicted_choice = 0
                elif data['preds'][1] > data['preds'][0]:
                    predicted_choice = 1
                else:
                    # Tie - default to choice 0
                    predicted_choice = 0
                
                # Check if correct (the choice with label=1 is the correct one)
                correct_choice = 0 if data['labels'][0] == 1 else 1
                if predicted_choice == correct_choice:
                    correct += 1
                total += 1
            
            return {'accuracy': correct / total if total > 0 else 0.0}
        
        # Special handling for SWAG - need to score choices (4-way)
        elif task_name == 'swag' and metadata and 'copa_id' in metadata:
            # Group predictions by swag_id (stored in copa_id field)
            swag_groups = defaultdict(lambda: {'preds': [], 'labels': [], 'choice_nums': []})
            for pred, label, swag_id, choice_num in zip(predictions, labels, 
                                                         metadata['copa_id'], metadata['choice_num']):
                swag_groups[swag_id]['preds'].append(pred)
                swag_groups[swag_id]['labels'].append(label)
                swag_groups[swag_id]['choice_nums'].append(choice_num)
            
            # For each SWAG question, pick choice with highest score/prediction
            correct = 0
            total = 0
            for swag_id, data in swag_groups.items():
                # Find the choice with highest prediction score
                predicted_choice = max(range(len(data['preds'])), key=lambda i: data['preds'][i])
                
                # Find the correct choice (the one with label=1)
                correct_choice = data['labels'].index(1) if 1 in data['labels'] else 0
                
                if predicted_choice == correct_choice:
                    correct += 1
                total += 1
            
            return {'accuracy': correct / total if total > 0 else 0.0}
        
        else:
            return {'accuracy': accuracy_score(labels, predictions)}
    
    elif metric_type == 'accuracy_and_f1':
        # CB: Both accuracy and macro-F1
        acc = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='macro')
        return {
            'accuracy': acc,
            'f1': f1,
            'avg': (acc + f1) / 2  # Official SuperGLUE uses average
        }
    
    elif metric_type == 'f1':
        num_labels = task_config['num_labels']
        average = 'binary' if num_labels == 2 else 'macro'
        result = {'f1': f1_score(labels, predictions, average=average)}
        if include_accuracy:
            result['accuracy'] = accuracy_score(labels, predictions)
        return result
    
    elif metric_type == 'f1a':
        # FIXED: Proper F1a computation for MultiRC
        if metadata is None or 'question_id' not in metadata:
            raise ValueError("F1a metric requires question_id in metadata")
        
        question_ids = metadata['question_id']
        f1a = compute_f1a_multirc(predictions, labels, question_ids)
        result = {'f1a': f1a}
        
        if include_accuracy:
            result['accuracy'] = accuracy_score(labels, predictions)
        
        return result
    
    elif metric_type == 'squad_f1_em':
        # FIXED: Proper F1 and EM computation for SQuAD v2
        # predictions: list of predicted answer strings
        # labels: list of ground truth answer dicts/strings
        ground_truths = [label['text'] if isinstance(label, dict) else label for label in labels]
        result = compute_squad_metrics(predictions, ground_truths)
        return result
    
    elif metric_type == 'matthews_corrcoef':
        metric = BinaryMatthewsCorrCoef()
        mcc = metric(torch.tensor(predictions), torch.tensor(labels))
        result = {'matthews_corrcoef': mcc.item()}
        if include_accuracy:
            result['accuracy'] = accuracy_score(labels, predictions)
        return result
    
    elif metric_type == 'pearson_spearman':
        pearson = pearsonr(predictions, labels)[0]
        spearman = spearmanr(predictions, labels)[0]
        result = {'spearman': spearman}
        if include_accuracy:
            result['pearson'] = pearson
            result['pearson_spearman'] = (pearson + spearman) / 2
        return result
    
    else:
        raise ValueError(f"Unknown metric type: {metric_type}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_task_info(task_name: str) -> Dict:
    """Get configuration for a task."""
    if task_name not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}")
    return TASK_CONFIGS[task_name]


if __name__ == '__main__':
    print("=" * 70)
    print("Benchmark Dataset Loader - Full Support")
    print("=" * 70)
    print("\nSupported Tasks:")
    print("\nGLUE (9 tasks):")
    print("  cola, sst2, mrpc, qqp, stsb, mnli, qnli, rte, wnli")
    print("\nAdvGLUE (6 adversarial tasks):")
    print("  adv_sst2, adv_qqp, adv_mnli, adv_mnli_mismatched, adv_qnli, adv_rte")
    print("\nPAWS (3 configs):")
    print("  paws (labeled_final), paws_swap (labeled_swap), paws_unlabeled")
    print("\nPAWS-X (7 languages):")
    print("  pawsx_en, pawsx_de, pawsx_es, pawsx_fr, pawsx_zh, pawsx_ja, pawsx_ko")
    print("\nHotpotQA (2 configs):")
    print("  hotpotqa (distractor), hotpotqa_fullwiki")
    print("\nANLI (3 rounds - Adversarial NLI):")
    print("  anli_r1, anli_r2, anli_r3")
    print("\nSWAG:")
    print("  swag (commonsense reasoning, 4-way multiple choice)")
    print("\nSuperGLUE (8 tasks + 2 diagnostic):")
    print("  boolq, cb, copa, multirc, record, wic, wsc")
    print("  axb, axg (diagnostic, test-only)")
    print("\nKey Features:")
    print("  ✓ MultiRC: F1a (per-question F1 average)")
    print("  ✓ CB: Accuracy + Macro-F1")
    print("  ✓ COPA: Proper choice scoring (2 examples per question)")
    print("  ✓ SWAG: 4-way multiple choice (4 examples per question)")
    print("  ✓ ReCoRD: Entity ranking (multiple examples per passage)")
    print("  ✓ PAWS/PAWS-X: Adversarial paraphrase detection with F1 metric")
    print("  ✓ HotpotQA: Multi-hop QA with F1/EM metrics (requires QA model)")
    print("  ✓ ANLI: Adversarial NLI with 3 rounds of increasing difficulty")
    print("  ✓ AdvGLUE: Adversarial versions of GLUE tasks")
    print("  ✓ All preprocessing functions implemented")
    print("  ✓ Test-only diagnostic tasks supported")
    print("=" * 70)
