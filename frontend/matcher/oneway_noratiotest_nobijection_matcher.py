"""One-way matcher w/o ratio-test, not enforcing bijection (i.e 1:1 matches).

Authors: Ayush Baid
"""
from frontend.matcher.generic_matcher import GenericMatcher


class OneWayNoRatioTestNoBijectionMatcher(GenericMatcher):
    """One-way matcher w/o ratio-test, not enforcing bijection."""

    def __init__(self):
        """Initializes the properties of the parent generic matcher."""
        super().__init__(
            ratio_test_threshold=None,
            is_mutual=False,
            is_bijection=False,
        )
