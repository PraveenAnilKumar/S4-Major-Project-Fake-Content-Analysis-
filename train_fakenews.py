#!/usr/bin/env python
"""
train_fakenews.py - Train Fake News Detection Model
Usage: python train_fakenews.py --dataset path/to/data.csv [options]
"""

import argparse
import os
import pandas as pd
from fake_news_detector import FakeNewsDetector

def main():
    parser = argparse.ArgumentParser(description='Train Fake News Detector')
    parser.add_argument('--dataset', required=True, help='Path to CSV file')
    parser.add_argument('--text-col', default='text', help='Column containing text (default: text)')
    parser.add_argument('--label-col', default='label', help='Column containing label (default: label)')
    parser.add_argument('--transformer', action='store_true', help='Use transformer model (DistilBERT)')
    parser.add_argument('--model-name', default='distilbert-base-uncased', help='Transformer model name')
    parser.add_argument('--save-path', default='models/fake_news/', help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs (for transformer)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (for transformer)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Validation split ratio (default: 0.2)')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of rows to load (for memory control)')
    args = parser.parse_args()

    print("🚀 train_fakenews.py started")
    print(f"📁 Dataset: {args.dataset}")
    if args.max_samples:
        print(f"📊 Using only first {args.max_samples} rows")
    print(f"📊 Test size: {args.test_size}")
    print(f"🤖 Use transformer: {args.transformer}")

    # Load only the required rows if max_samples is set
    if args.max_samples:
        df = pd.read_csv(args.dataset, nrows=args.max_samples)
        # Save temporary file with sampled data
        temp_file = "temp_sample.csv"
        df.to_csv(temp_file, index=False)
        dataset_path = temp_file
    else:
        dataset_path = args.dataset

    detector = FakeNewsDetector(use_transformer=args.transformer, model_name=args.model_name)
    print("✅ FakeNewsDetector created")

    print("🔧 Starting training...")
    detector.train(
        csv_path=dataset_path,
        text_column=args.text_col,
        label_column=args.label_col,
        test_size=args.test_size,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    print("🏁 Training finished!")

    # Clean up temporary file if created
    if args.max_samples and os.path.exists(temp_file):
        os.remove(temp_file)

if __name__ == "__main__":
    main()#!/usr/bin/env python
"""
train_fakenews.py - Train Fake News Detection Model
Usage: python train_fakenews.py --dataset path/to/data.csv [options]
"""

import argparse
import os
import pandas as pd
from fake_news_detector import FakeNewsDetector

def main():
    parser = argparse.ArgumentParser(description='Train Fake News Detector')
    parser.add_argument('--dataset', required=True, help='Path to CSV file')
    parser.add_argument('--text-col', default='text', help='Column containing text (default: text)')
    parser.add_argument('--label-col', default='label', help='Column containing label (default: label)')
    parser.add_argument('--transformer', action='store_true', help='Use transformer model (DistilBERT)')
    parser.add_argument('--model-name', default='distilbert-base-uncased', help='Transformer model name')
    parser.add_argument('--save-path', default='models/fake_news/', help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs (for transformer)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (for transformer)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Validation split ratio (default: 0.2)')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of rows to load (for memory control)')
    args = parser.parse_args()

    print("🚀 train_fakenews.py started")
    print(f"📁 Dataset: {args.dataset}")
    if args.max_samples:
        print(f"📊 Using only first {args.max_samples} rows")
    print(f"📊 Test size: {args.test_size}")
    print(f"🤖 Use transformer: {args.transformer}")

    # Load only the required rows if max_samples is set
    if args.max_samples:
        df = pd.read_csv(args.dataset, nrows=args.max_samples)
        # Save temporary file with sampled data
        temp_file = "temp_sample.csv"
        df.to_csv(temp_file, index=False)
        dataset_path = temp_file
    else:
        dataset_path = args.dataset

    detector = FakeNewsDetector(use_transformer=args.transformer, model_name=args.model_name)
    print("✅ FakeNewsDetector created")

    print("🔧 Starting training...")
    detector.train(
        csv_path=dataset_path,
        text_column=args.text_col,
        label_column=args.label_col,
        test_size=args.test_size,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    print("🏁 Training finished!")

    # Clean up temporary file if created
    if args.max_samples and os.path.exists(temp_file):
        os.remove(temp_file)

if __name__ == "__main__":
    main()#!/usr/bin/env python
"""
train_fakenews.py - Train Fake News Detection Model
Usage: python train_fakenews.py --dataset path/to/data.csv [options]
"""

import argparse
import os
import pandas as pd
from fake_news_detector import FakeNewsDetector

def main():
    parser = argparse.ArgumentParser(description='Train Fake News Detector')
    parser.add_argument('--dataset', required=True, help='Path to CSV file')
    parser.add_argument('--text-col', default='text', help='Column containing text (default: text)')
    parser.add_argument('--label-col', default='label', help='Column containing label (default: label)')
    parser.add_argument('--transformer', action='store_true', help='Use transformer model (DistilBERT)')
    parser.add_argument('--model-name', default='distilbert-base-uncased', help='Transformer model name')
    parser.add_argument('--save-path', default='models/fake_news/', help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=3, help='Number of epochs (for transformer)')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size (for transformer)')
    parser.add_argument('--test-size', type=float, default=0.2, help='Validation split ratio (default: 0.2)')
    parser.add_argument('--max-samples', type=int, default=None, help='Maximum number of rows to load (for memory control)')
    args = parser.parse_args()

    print("🚀 train_fakenews.py started")
    print(f"📁 Dataset: {args.dataset}")
    if args.max_samples:
        print(f"📊 Using only first {args.max_samples} rows")
    print(f"📊 Test size: {args.test_size}")
    print(f"🤖 Use transformer: {args.transformer}")

    # Load only the required rows if max_samples is set
    if args.max_samples:
        df = pd.read_csv(args.dataset, nrows=args.max_samples)
        # Save temporary file with sampled data
        temp_file = "temp_sample.csv"
        df.to_csv(temp_file, index=False)
        dataset_path = temp_file
    else:
        dataset_path = args.dataset

    detector = FakeNewsDetector(use_transformer=args.transformer, model_name=args.model_name)
    print("✅ FakeNewsDetector created")

    print("🔧 Starting training...")
    detector.train(
        csv_path=dataset_path,
        text_column=args.text_col,
        label_column=args.label_col,
        test_size=args.test_size,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    print("🏁 Training finished!")

    # Clean up temporary file if created
    if args.max_samples and os.path.exists(temp_file):
        os.remove(temp_file)

if __name__ == "__main__":
    main()