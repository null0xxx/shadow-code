"""Tests for skills.py -- skill registry and built-in skills."""

import unittest

from shadow_code.skills import (
    _BILINGUAL_PREAMBLE,
    _SKILLS,
    get_skill,
    list_skills,
    register_skill,
)


class TestSkillRegistry(unittest.TestCase):
    """Tests for the skill registry functions."""

    def test_builtin_skills_registered(self):
        """All 13 built-in skills should be registered on import."""
        expected = {
            "commit",
            "simplify",
            "review",
            "explain",
            "debug",
            "stuck",
            "pr",
            "remember",
            "verify",
            "init",
            "test",
            "refactor",
            "search",
        }
        actual = set(_SKILLS.keys())
        self.assertTrue(expected.issubset(actual), f"Missing: {expected - actual}")

    def test_get_skill_exists(self):
        result = get_skill("commit")
        self.assertIsNotNone(result)
        desc, prompt = result
        self.assertIsInstance(desc, str)
        self.assertIsInstance(prompt, str)
        self.assertIn("git commit", desc.lower())

    def test_get_skill_not_found(self):
        result = get_skill("nonexistent_skill_xyz")
        self.assertIsNone(result)

    def test_list_skills_returns_sorted(self):
        skills = list_skills()
        self.assertIsInstance(skills, list)
        self.assertGreater(len(skills), 0)
        names = [name for name, _desc in skills]
        self.assertEqual(names, sorted(names))

    def test_list_skills_format(self):
        skills = list_skills()
        for name, desc in skills:
            self.assertIsInstance(name, str)
            self.assertIsInstance(desc, str)
            self.assertGreater(len(name), 0)
            self.assertGreater(len(desc), 0)

    def test_register_custom_skill(self):
        register_skill("_test_custom", "A test skill", "Do a test thing")
        result = get_skill("_test_custom")
        self.assertIsNotNone(result)
        desc, prompt = result
        self.assertEqual(desc, "A test skill")
        self.assertEqual(prompt, _BILINGUAL_PREAMBLE + "Do a test thing")
        # Cleanup
        del _SKILLS["_test_custom"]

    def test_register_overwrites(self):
        register_skill("_test_overwrite", "v1", "prompt v1")
        register_skill("_test_overwrite", "v2", "prompt v2")
        desc, prompt = get_skill("_test_overwrite")
        self.assertEqual(desc, "v2")
        self.assertEqual(prompt, _BILINGUAL_PREAMBLE + "prompt v2")
        del _SKILLS["_test_overwrite"]


class TestBuiltinSkillContent(unittest.TestCase):
    """Tests for built-in skill prompt content."""

    def test_commit_skill_has_safety_rules(self):
        _, prompt = get_skill("commit")
        self.assertIn("--no-verify", prompt)
        self.assertIn("amend", prompt.lower())

    def test_review_skill_mentions_security(self):
        _, prompt = get_skill("review")
        self.assertIn("## SECURITY", prompt)
        self.assertIn("common-security", prompt)

    def test_debug_skill_mentions_root_cause(self):
        _, prompt = get_skill("debug")
        self.assertIn("root cause", prompt)

    def test_verify_skill_requires_execution(self):
        _, prompt = get_skill("verify")
        self.assertIn("Actually run", prompt)

    def test_each_skill_has_nonempty_prompt(self):
        for name, (_desc, prompt) in _SKILLS.items():
            self.assertGreater(len(prompt), 20, f"Skill '{name}' has too-short prompt")


if __name__ == "__main__":
    unittest.main()
