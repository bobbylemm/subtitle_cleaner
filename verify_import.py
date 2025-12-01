import sys
import os
sys.path.append(os.getcwd())
try:
    from app.services.universal_corrector import UniversalCorrectionService
    print("Import successful!")
except Exception as e:
    print(f"Import failed: {e}")
    sys.exit(1)
