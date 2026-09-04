import pytest
# Import your class from your actual file
from tagalog_to_baybayin import TagalogToBaybayin

@pytest.fixture
def engine():
    """Initializes the TagalogToBaybayin class for testing"""
    return TagalogToBaybayin()

# These are the 10 cases that gave you 100% coverage in your screenshot
@pytest.mark.parametrize("input_text, expected, description", [
    ("", "", "Empty input guard clause"),
    ("mga", "ᜋᜄ", "Linguistic exception path"),
    ("a i u", "ᜀ ᜁ ᜂ", "Standalone vowel glyphs"),
    ("Ilog", "ᜁᜎᜓᜄ᜔", "Standalone vowel at word start"), # Ensure only ONE comma here
    ("Baya", "ᜊᜌ", "CV pair with inherent 'a'"),
    ("Ngiti", "ᜅᜒᜆᜒ", "CV pair with upper kudlit (e/i)"),
    ("Opo", "ᜂᜉᜓ", "CV pair with lower kudlit (o/u)"),
    ("Salamat", "ᜐᜎᜋᜆ᜔", "Final consonant with virama"),
    ("Ang", "ᜀᜅ᜔", "Final digraph consonant logic"),
    ("A B", "ᜀ ᜊ᜔", "Space preservation between tokens"),
])
def test_white_box_coverage(engine, input_text, expected, description):
    """
    Test Case: Validates if the TTB engine correctly applies 
    the Lopez Method and Regex tokenization.
    """
    result = engine.translate(input_text)
    
    # This is the actual 'test'—checking if your code's output matches the 'Ground Truth'
    assert result == expected