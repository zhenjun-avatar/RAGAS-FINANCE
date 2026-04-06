"""Lead extraction tools.

Currently includes:
- Bocha web-search / AI-search wrapper
- Part-time graduate admissions (在职研究生) high-intent lead extraction
"""

from .part_time_graduate_leads import extract_part_time_graduate_leads

__all__ = ["extract_part_time_graduate_leads"]

