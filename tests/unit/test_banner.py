"""Tests for banner.py -- startup wordmark, gradient, and tips block."""

import unittest

from shadow_code import banner
from shadow_code.theme import THEME


class TestBannerArt(unittest.TestCase):
    def test_art_is_bounded_and_ascii(self):
        self.assertLessEqual(len(banner.BANNER_ART), 8)
        self.assertGreaterEqual(len(banner.BANNER_ART), 4)
        for line in banner.BANNER_ART:
            self.assertTrue(line.isascii())
            self.assertEqual(len(line), len(line.rstrip()))  # no trailing spaces
        self.assertEqual(banner.BANNER_WIDTH, max(len(line) for line in banner.BANNER_ART))

    def test_art_spells_shadow_code(self):
        # Every letter slot contributes ink to every row band.
        joined = "\n".join(banner.BANNER_ART)
        self.assertIn("#####", joined)  # H/E crossbars
        self.assertNotIn("\t", joined)

    def test_narrow_terminal_falls_back_to_plain_title(self):
        self.assertEqual(banner.banner_lines(banner.BANNER_WIDTH - 1), ["SHADOW CODE"])
        self.assertEqual(banner.banner_lines(banner.BANNER_WIDTH), list(banner.BANNER_ART))
        self.assertEqual(banner.banner_lines(None), list(banner.BANNER_ART))


class TestTips(unittest.TestCase):
    def test_tips_block_content(self):
        lines = banner.tips_lines()
        self.assertEqual(lines[0], "Tips for getting started:")
        self.assertEqual(len(lines), 4)
        self.assertTrue(lines[1].startswith("  1. "))
        self.assertIn("/help", lines[3])


class TestGradient(unittest.TestCase):
    def test_endpoints_reuse_theme_palette(self):
        self.assertEqual(banner.gradient_color(0.0), THEME.accent)
        self.assertEqual(banner.gradient_color(1.0), THEME.highlight)

    def test_midpoint_approaches_info(self):
        self.assertEqual(banner.gradient_color(0.5), THEME.info)

    def test_clamps_out_of_range(self):
        self.assertEqual(banner.gradient_color(-1), THEME.accent)
        self.assertEqual(banner.gradient_color(2), THEME.highlight)

    def test_monotonic_hue_drift(self):
        # A gradient, not a flat fill: many distinct colors across the width.
        colors = {banner.column_color(i) for i in range(banner.BANNER_WIDTH)}
        self.assertGreater(len(colors), banner.BANNER_WIDTH // 3)


class TestStyledFragments(unittest.TestCase):
    def test_banner_fragments_gradient_when_colored(self):
        fragments = banner.styled_banner_fragments(colors=True)
        hex_styles = {style for style, _text in fragments if style.startswith("#")}
        self.assertGreater(len(hex_styles), 10)
        # Spaces never carry a style (terminal background shows through).
        for style, text in fragments:
            if text.strip() == "":
                self.assertEqual(style, "")

    def test_banner_fragments_plain_without_colors(self):
        fragments = banner.styled_banner_fragments(colors=False)
        self.assertTrue(all(style == "" for style, _text in fragments))
        plain = "".join(text for _style, text in fragments)
        self.assertEqual(plain, "\n".join(banner.BANNER_ART))

    def test_banner_fragments_narrow_fallback(self):
        fragments = banner.styled_banner_fragments(colors=True, width=10)
        plain = "".join(text for _style, text in fragments)
        self.assertEqual(plain, "SHADOW CODE")

    def test_tips_fragments_accent_help_when_colored(self):
        fragments = banner.styled_tips_fragments(colors=True)
        self.assertIn(("class:tips-accent", "/help"), fragments)
        plain = "".join(text for _style, text in fragments)
        self.assertEqual(plain, "\n".join(banner.tips_lines()))

    def test_tips_fragments_plain_without_colors(self):
        fragments = banner.styled_tips_fragments(colors=False)
        self.assertTrue(all(style == "" for style, _text in fragments))


if __name__ == "__main__":
    unittest.main()
