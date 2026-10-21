import random
from pathlib import Path


def shuffle_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    lines = [line.strip() for line in lines if line.strip()]
    
    random.shuffle(lines)
    
    with open(file_path, 'w') as f:
        for line in lines:
            f.write(line + '\n')
    
    print(f"Shuffled {file_path}: {len(lines)} lines")


if __name__ == '__main__':
    base_dir = Path(__file__).parent.parent / 'raw_data' / 'finedance'
    
    files = [
        base_dir / 'processed_test.txt',
        base_dir / 'processed_train.txt',
        base_dir / 'processed_val.txt'
    ]
    
    random.seed(42)
    
    for file_path in files:
        if file_path.exists():
            shuffle_file(file_path)
        else:
            print(f"File not found: {file_path}")
    
    print("\nAll files shuffled successfully!")
