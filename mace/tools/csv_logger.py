import os
import csv
import time
from collections import defaultdict
import logging

class CSVLogger:
    def __init__(self, job_name, base_dir=None, mode='a'):
        if base_dir is None:
            base_dir = os.getcwd()
        self.job_dir = os.path.join(base_dir, job_name)
        self.mode = mode
        self.writers = {}
        self.files = {}
        self.headers = {}
        self.file_paths = {}  # Track file paths for each category
        os.makedirs(self.job_dir, exist_ok=True)
        logging.info(f"CSVLogger initialized. Logs will be saved to: {self.job_dir} (mode={mode})")

        # If mode is 'w', we should backup existing logs or just be ready to overwrite.
        # Since we create files lazily in _write_to_csv, we handle the 'w' logic there or here.
        # To strictly follow 'w', we might want to clear the directory or delete specific files.
        # But since we have subfolders (train, val), it's complex. 
        # Let's handle it by deleting the specific log file when we first open it in _write_to_csv if mode is 'w'.
        self.files_initialized = set()

    def log(self, metrics, step):
        # Group metrics by category (prefix)
        grouped_metrics = defaultdict(dict)
        
        for key, value in metrics.items():
            if "/" in key:
                prefix, suffix = key.split("/", 1)
                grouped_metrics[prefix][suffix] = value
            else:
                # Default to 'train' for keys without prefix (like 'lr', 'warmup_done')
                grouped_metrics["train"][key] = value
        
        for category, cat_metrics in grouped_metrics.items():
            self._write_to_csv(category, cat_metrics, step)

    def _expand_csv_headers(self, category, new_keys):
        """Expand CSV file with new columns by rewriting with updated headers."""
        file_path = self.file_paths[category]
        
        # Close current writer and file
        if category in self.files:
            self.files[category].close()
        
        # Read all existing data
        existing_rows = []
        if os.path.exists(file_path):
            with open(file_path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_rows = list(reader)
        
        # Update headers with new keys
        old_headers = self.headers[category]
        new_headers = old_headers + sorted(new_keys)
        self.headers[category] = new_headers
        
        # Rewrite file with expanded headers
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=new_headers, extrasaction='ignore')
            writer.writeheader()
            for row in existing_rows:
                writer.writerow(row)
        
        # Reopen file in append mode
        f = open(file_path, "a", newline="", encoding="utf-8")
        self.files[category] = f
        self.writers[category] = csv.DictWriter(f, fieldnames=new_headers, extrasaction='ignore')
        
        logging.info(f"Expanded CSV headers for {category} with new keys: {new_keys}")

    def _write_to_csv(self, category, metrics, step):
        dir_path = os.path.join(self.job_dir, category)
        file_path = os.path.join(dir_path, "log.csv")
        
        # Add step and timestamp
        row = {"global_step": step, "timestamp": time.time(), **metrics}
        
        if category not in self.writers:
            os.makedirs(dir_path, exist_ok=True)
            self.file_paths[category] = file_path
            
            # Handle mode='w' (overwrite) - only do this once per file per run
            if self.mode == 'w' and file_path not in self.files_initialized:
                if os.path.exists(file_path):
                    try:
                        # Backup existing file
                        import shutil
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        backup_path = f"{file_path}.{timestamp}.bak"
                        shutil.move(file_path, backup_path)
                        logging.info(f"Backed up existing log to {backup_path}")
                    except OSError as e:
                        logging.warning(f"Failed to backup {file_path}: {e}")
                self.files_initialized.add(file_path)

            # Check if file exists to determine if we need header
            file_exists = os.path.exists(file_path)
            
            # If file exists, try to read header to see if we have new keys
            existing_fieldnames = []
            if file_exists:
                with open(file_path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    try:
                        existing_fieldnames = next(reader)
                    except StopIteration:
                        pass
            
            # If we have new keys that aren't in existing header, or if it's a new file
            current_keys = list(row.keys())
            
            # Merge existing fieldnames with current keys
            fieldnames = existing_fieldnames if existing_fieldnames else ["global_step", "timestamp"]
            for k in current_keys:
                if k not in fieldnames:
                    fieldnames.append(k)
            
            self.headers[category] = fieldnames
            
            # Open file in append mode
            f = open(file_path, "a", newline="", encoding="utf-8")
            self.files[category] = f
            self.writers[category] = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            
            # If it's a new file or we expanded headers (and file was empty), write header
            if not file_exists or not existing_fieldnames:
                self.writers[category].writeheader()
        
        # Check for new keys and expand headers if needed
        current_keys = set(row.keys())
        known_keys = set(self.headers[category])
        new_keys = current_keys - known_keys
        
        if new_keys:
            # Expand CSV with new columns by rewriting the file
            self._expand_csv_headers(category, new_keys)
             
        self.writers[category].writerow(row)
        self.files[category].flush()

    def close(self):
        for f in self.files.values():
            f.close()
