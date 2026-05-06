import textwrap
import unittest
from pathlib import Path

from pipeline.stage1_parse import parse_file
from pipeline.stage2_resolve import resolve_records


class EtlRegressionTests(unittest.TestCase):
    def parse(self, raw: str, rel_path: str = "en/source.md") -> list[dict]:
        return parse_file(rel_path, textwrap.dedent(raw).strip() + "\n", {})

    def test_indented_mdx_wrapper_content_is_not_blank_language_code(self) -> None:
        records = self.parse(
            """
            # Source

            ## Setup

            <Steps>
              <Step title="Run it">
                Intro prose before code.

                <CodeGroup>
                  ```python Python theme={null}
                  print("py")
                  ```

                  ```typescript TypeScript theme={null}
                  console.log("ts")
                  ```
                </CodeGroup>

                After prose.

                ```bash theme={null}
                echo ok
                ```
              </Step>
            </Steps>
            """
        )

        codeblocks = [record for record in records if record["kind"] == "codeblock"]
        self.assertEqual(["python", "typescript", "bash"], [block["language"] for block in codeblocks])
        self.assertFalse(any("<!--codegroup:" in block["text"] for block in codeblocks))
        self.assertFalse(any("```" in block["text"] for block in codeblocks))

        setup = next(record for record in records if record["kind"] == "section")
        self.assertIn("Intro prose before code.", setup["text"])
        self.assertIn("After prose.", setup["text"])

    def test_indented_mdx_table_remains_a_table_row(self) -> None:
        records = self.parse(
            """
            # Source

            ## Quick reference

            <Steps>
              <Step title="Choose">
                | Option | Description |
                | --- | --- |
                | A | Alpha |
              </Step>
            </Steps>
            """
        )

        self.assertEqual([], [record for record in records if record["kind"] == "codeblock"])
        rows = [record for record in records if record["kind"] == "tablerow"]
        self.assertEqual(1, len(rows))
        self.assertEqual(["A", "Alpha"], rows[0]["cells"])

    def test_top_level_content_before_first_h2_goes_to_overview(self) -> None:
        records = self.parse(
            """
            # Source

            Top level intro with a [target](/en/target).

            ```bash theme={null}
            pwd
            ```

            ## Next

            Later text.
            """
        )

        sections = [record for record in records if record["kind"] == "section"]
        overview = next(section for section in sections if section["anchor"] == "overview")
        self.assertIn("Top level intro", overview["text"])

        codeblocks = [record for record in records if record["kind"] == "codeblock"]
        self.assertEqual(1, len(codeblocks))
        self.assertEqual("bash", codeblocks[0]["language"])
        self.assertEqual(overview["id"], codeblocks[0]["section_id"])

    def test_exported_jsx_component_with_destructured_params_is_stripped(self) -> None:
        records = self.parse(
            """
            # Source

            export const Widget = ({platform = "all"}) => {
              const values = {
                one: "two"
              };
              return <div>{values.one}</div>;
            };

            ## Real content

            Keep this paragraph.
            """
        )

        codeblocks = [record for record in records if record["kind"] == "codeblock"]
        self.assertEqual([], codeblocks)
        sections = [record for record in records if record["kind"] == "section"]
        self.assertEqual(["real-content"], [section["anchor"] for section in sections])
        self.assertIn("Keep this paragraph.", sections[0]["text"])

    def test_exported_jsx_component_does_not_swallow_following_markdown(self) -> None:
        records = self.parse(
            """
            # Source

            export const Widget = () => {
              return <>
                {items.map(item => <span>{item.label}</span>)}
              </>;
            };

            ## Application data

            Keep this paragraph.
            """
        )

        sections = [record for record in records if record["kind"] == "section"]
        self.assertEqual(["application-data"], [section["anchor"] for section in sections])
        self.assertIn("Keep this paragraph.", sections[0]["text"])

    def test_large_exported_jsx_component_preserves_following_markdown(self) -> None:
        source = Path("documents/en/claude-directory.md").read_text(encoding="utf-8")
        records = parse_file("en/claude-directory.md", source, {})
        anchors = [record["anchor"] for record in records if record["kind"] == "section"]

        self.assertIn("application-data", anchors)

    def test_indented_prose_continuation_is_not_codeblock(self) -> None:
        records = self.parse(
            """
            # Source

            ## Settings files

            * **Managed settings**: centralized control.

              * **File-based**: deployed to system directories:

                * macOS: `/Library/Application Support/ClaudeCode/`
                * Linux and WSL: `/etc/claude-code/`

                <Warning>
                  The legacy path is no longer supported.
                </Warning>

                File-based managed settings also support a drop-in directory at `managed-settings.d/`.

                Following the systemd convention, base files are merged first.
            """
        )

        codeblocks = [record for record in records if record["kind"] == "codeblock"]
        self.assertEqual([], codeblocks)
        section = next(record for record in records if record["kind"] == "section")
        self.assertIn("File-based managed settings also support", section["text"])
        self.assertIn("Following the systemd convention", section["text"])

    def test_empty_separator_heading_is_omitted(self) -> None:
        records = self.parse(
            """
            # Source

            ## Setup

            ## Quick setup

            Keep this paragraph.
            """
        )

        sections = [record for record in records if record["kind"] == "section"]
        self.assertEqual(["quick-setup"], [section["anchor"] for section in sections])

    def test_navigation_table_contributes_section_text_without_table_rows(self) -> None:
        records = self.parse(
            """
            # Source

            ## Quick reference

            | If you want to... | Do this |
            | --- | --- |
            | Define a tool | See [Create a custom tool](#create-a-custom-tool). |

            ## Create a custom tool

            Target text.
            """
        )

        quick_reference = next(
            record
            for record in records
            if record["kind"] == "section" and record["anchor"] == "quick-reference"
        )
        self.assertIn("Define a tool", quick_reference["text"])
        self.assertEqual([], [record for record in records if record["kind"] == "tablerow"])

    def test_links_to_self_loop_is_not_emitted(self) -> None:
        records_by_file = {
            Path("source.jsonl"): self.parse(
                """
                # Source

                ## Resume subagents

                You can [resume a subagent](#resume-subagents).
                """
            )
        }

        resolved = resolve_records(records_by_file)
        relationships = [
            record
            for records in resolved.values()
            for record in records
            if record["kind"] == "relationship"
        ]
        self.assertFalse(
            any(
                relationship["type"] == "LINKS_TO"
                and relationship["source_id"] == relationship["target_id"]
                for relationship in relationships
            )
        )

    def test_anchor_resolution_matches_punctuation_variants(self) -> None:
        records_by_file = {
            Path("source.jsonl"): self.parse(
                """
                # Source

                ## Overview

                See [CLAUDE.md files](/en/target#claude-md-files).
                """
            ),
            Path("target.jsonl"): self.parse(
                """
                # Target

                ## CLAUDE.md files

                Target text.
                """,
                rel_path="en/target.md",
            ),
        }

        resolved = resolve_records(records_by_file)
        unresolved = [
            record
            for records in resolved.values()
            for record in records
            if record["kind"] == "unresolved_link"
        ]
        relationships = [
            record
            for records in resolved.values()
            for record in records
            if record["kind"] == "relationship"
        ]

        self.assertEqual([], unresolved)
        self.assertTrue(any(record["type"] == "LINKS_TO" for record in relationships))

    def test_h5_heading_can_be_link_target(self) -> None:
        records_by_file = {
            Path("source.jsonl"): self.parse(
                """
                # Source

                ## Overview

                See [details](/en/target#deep-detail).
                """
            ),
            Path("target.jsonl"): self.parse(
                """
                # Target

                ## Parent

                #### Child

                ##### Deep detail

                Target text.
                """,
                rel_path="en/target.md",
            ),
        }

        resolved = resolve_records(records_by_file)
        unresolved = [
            record
            for records in resolved.values()
            for record in records
            if record["kind"] == "unresolved_link"
        ]

        self.assertEqual([], unresolved)

    def test_dom_anchors_are_ignored(self) -> None:
        records_by_file = {
            Path("source.jsonl"): self.parse(
                """
                # Source

                ## File reference

                See [`CLAUDE.md`](#ce-claude-md).
                """
            )
        }

        resolved = resolve_records(records_by_file)
        unresolved = [
            record
            for records in resolved.values()
            for record in records
            if record["kind"] == "unresolved_link"
        ]
        relationships = [
            record
            for records in resolved.values()
            for record in records
            if record["kind"] == "relationship" and record["type"] == "LINKS_TO"
        ]

        self.assertEqual([], unresolved)
        self.assertEqual([], relationships)

    def test_encoded_template_https_href_is_external(self) -> None:
        records_by_file = {
            Path("source.jsonl"): self.parse(
                """
                # Source

                ## Overview

                [Contact sales](%60https://claude.com/contact-sales?$%7Butm('contact_sales')%7D%60)
                """
            )
        }

        resolved = resolve_records(records_by_file)
        unresolved = [
            record
            for records in resolved.values()
            for record in records
            if record["kind"] == "unresolved_link"
        ]

        self.assertEqual([], unresolved)

    def test_known_legacy_page_alias_resolves(self) -> None:
        records_by_file = {
            Path("source.jsonl"): self.parse(
                """
                # Source

                ## Overview

                See [old Foundry URL](/en/azure-ai-foundry).
                """
            ),
            Path("target.jsonl"): self.parse(
                """
                # Microsoft Foundry

                ## Prerequisites

                Target text.
                """,
                rel_path="en/microsoft-foundry.md",
            ),
        }

        resolved = resolve_records(records_by_file)
        unresolved = [
            record
            for records in resolved.values()
            for record in records
            if record["kind"] == "unresolved_link"
        ]
        relationships = [
            record
            for records in resolved.values()
            for record in records
            if record["kind"] == "relationship"
        ]

        self.assertEqual([], unresolved)
        self.assertTrue(any(record["type"] == "LINKS_TO_PAGE" for record in relationships))


if __name__ == "__main__":
    unittest.main()
