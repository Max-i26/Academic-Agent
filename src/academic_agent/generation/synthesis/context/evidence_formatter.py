"""Evidence formatter providing secure XML escaping for untrusted reference data."""

import html
from academic_agent.generation.synthesis.models import EvidenceContext, GroundedEvidence
from academic_agent.generation.synthesis.protocols import EvidenceFormatterProtocol


class EvidenceFormatter(EvidenceFormatterProtocol):
    """Formats retrieved academic evidence into safely delimited and escaped XML structures.

    Protects against closing-tag breakout and attribute injection while preserving mathematical
    notation, Greek letters, and LaTeX equations.
    """

    @staticmethod
    def escape_text(text: str) -> str:
        """Escape XML/HTML characters in text content (&, <, >, ", ')."""
        return html.escape(text, quote=True)

    @staticmethod
    def escape_attribute(value: str) -> str:
        """Escape XML/HTML characters inside attribute quotes (&, <, >, ", ')."""
        return html.escape(str(value), quote=True)

    def format_item(self, item: GroundedEvidence) -> str:
        """Render a single GroundedEvidence passage with escaped attributes and content."""
        esc_id = self.escape_attribute(item.evidence_id)
        esc_source = self.escape_attribute(item.source_file)

        sorted_pages = sorted(set(item.source_pages))
        if not sorted_pages:
            pages_str = "none"
        elif len(sorted_pages) == 1:
            pages_str = str(sorted_pages[0])
        elif sorted_pages[-1] - sorted_pages[0] == len(sorted_pages) - 1:
            pages_str = f"{sorted_pages[0]}-{sorted_pages[-1]}"
        else:
            pages_str = ", ".join(str(p) for p in sorted_pages)

        esc_pages = self.escape_attribute(pages_str)
        esc_content = self.escape_text(item.text.strip())

        return f'  <evidence id="{esc_id}" source="{esc_source}" pages="{esc_pages}">\n    {esc_content}\n  </evidence>'

    def format_evidence_block(self, context: EvidenceContext) -> str:
        """Format an entire EvidenceContext into a structured <evidence_package> XML block."""
        count = len(context.items)
        if count == 0:
            return '<evidence_package count="0">\n</evidence_package>'

        items_formatted = [self.format_item(item) for item in context.items]
        body = "\n".join(items_formatted)
        return f'<evidence_package count="{count}">\n{body}\n</evidence_package>'
