import json
from pathlib import Path
from typing import Dict, Tuple, List

class INCARValidator:
    def __init__(self, schema_path: str = "database/INCAR_tags/incar_schema.json"):
        self.schema_path = Path(schema_path)
        with open(self.schema_path, 'r', encoding='utf-8') as f:
            self.schema = json.load(f)
    
    def validate_tag(self, tag: str, value: str) -> Tuple[bool, str]:
        """Validate a single INCAR tag and its value."""
        if tag not in self.schema:
            return False, f"Unknown INCAR tag: {tag}"
        
        tag_info = self.schema[tag]
        
        # Validate based on type
        if tag_info["type"] == "boolean":
            if value.upper() not in [".TRUE.", ".FALSE."]:
                return False, f"Invalid boolean value for {tag}. Must be .TRUE. or .FALSE."
        
        elif tag_info["type"] == "numeric":
            try:
                float(value)
            except ValueError:
                return False, f"Invalid numeric value for {tag}"
        
        elif tag_info["valid_values"]:
            if value not in tag_info["valid_values"]:
                return False, f"Invalid value for {tag}. Valid values are: {', '.join(tag_info['valid_values'])}"
        
        return True, "Valid value"

    def validate_incar(self, incar_content: str) -> List[Dict[str, str]]:
        """Validate entire INCAR file content."""
        validation_results = []
        
        lines = incar_content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                tag, value = [part.strip() for part in line.split('=', 1)]
                is_valid, message = self.validate_tag(tag, value)
                validation_results.append({
                    'tag': tag,
                    'value': value,
                    'is_valid': is_valid,
                    'message': message
                })
            except ValueError:
                validation_results.append({
                    'tag': line,
                    'value': '',
                    'is_valid': False,
                    'message': 'Invalid INCAR line format'
                })
        
        return validation_results