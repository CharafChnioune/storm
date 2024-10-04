"""https://github.com/arnaudmiribel/stoc"""

import re
import streamlit as st
import unidecode

# CSS om links in de inhoudsopgave te verbergen
DISABLE_LINK_CSS = """
<style>
a.toc {
    color: inherit;
    text-decoration: none; /* geen onderstreping */
}
</style>"""


class stoc:
    def __init__(self):
        # Lijst om inhoudsopgave-items op te slaan
        self.toc_items = list()

    def h1(self, text: str, write: bool = True):
        # Voeg een h1 kop toe aan de pagina en de inhoudsopgave
        if write:
            st.write(f"# {text}")
        self.toc_items.append(("h1", text))

    def h2(self, text: str, write: bool = True):
        # Voeg een h2 kop toe aan de pagina en de inhoudsopgave
        if write:
            st.write(f"## {text}")
        self.toc_items.append(("h2", text))

    def h3(self, text: str, write: bool = True):
        # Voeg een h3 kop toe aan de pagina en de inhoudsopgave
        if write:
            st.write(f"### {text}")
        self.toc_items.append(("h3", text))

    def toc(self, expander):
        # Genereer en toon de inhoudsopgave
        st.write(DISABLE_LINK_CSS, unsafe_allow_html=True)
        if expander is None:
            expander = st.sidebar.expander("**Inhoudsopgave**", expanded=True)
        with expander:
            with st.container(height=600, border=False):
                markdown_toc = ""
                for title_size, title in self.toc_items:
                    h = int(title_size.replace("h", ""))
                    markdown_toc += (
                            " " * 2 * h
                            + "- "
                            + f'<a href="#{normalize(title)}" class="toc"> {title}</a> \n'
                    )
                st.write(markdown_toc, unsafe_allow_html=True)

    @classmethod
    def get_toc(cls, markdown_text: str, topic=""):
        def increase_heading_depth_and_add_top_heading(markdown_text, new_top_heading):
            # Verhoog de diepte van alle koppen en voeg een nieuwe topkop toe
            lines = markdown_text.splitlines()
            increased_depth_lines = ['#' + line if line.startswith('#') else line for line in lines]
            increased_depth_lines.insert(0, f"# {new_top_heading}")
            return '\n'.join(increased_depth_lines)

        if topic:
            markdown_text = increase_heading_depth_and_add_top_heading(markdown_text, topic)
        
        # Genereer inhoudsopgave uit markdown tekst
        toc = []
        for line in markdown_text.splitlines():
            if line.startswith('#'):
                heading_text = line.lstrip('#').strip()
                slug = re.sub(r'[^a-zA-Z0-9\s-]', '', heading_text).lower().replace(' ', '-')
                level = line.count('#') - 1
                toc.append('  ' * level + f'- [{heading_text}](#{slug})')
        return '\n'.join(toc)

    @classmethod
    def from_markdown(cls, text: str, expander=None):
        # Maak een inhoudsopgave van markdown tekst
        self = cls()
        for line in text.splitlines():
            if line.startswith("###"):
                self.h3(line[3:], write=False)
            elif line.startswith("##"):
                self.h2(line[2:], write=False)
            elif line.startswith("#"):
                self.h1(line[1:], write=False)
        
        # Pas lettergrootte aan voor verschillende elementen
        custom_css = """
        <style>
            /* Pas de lettergrootte aan voor koppen */
            h1 { font-size: 28px; }
            h2 { font-size: 24px; }
            h3 { font-size: 22px; }
            h4 { font-size: 20px; }
            h5 { font-size: 18px; }
            /* Pas de lettergrootte aan voor normale tekst */
            p { font-size: 18px; }
        </style>
        """
        st.markdown(custom_css, unsafe_allow_html=True)

        st.write(text)
        self.toc(expander=expander)


def normalize(s):
    """
    Normaliseer titels als geldige HTML-id's voor ankers
    >>> normalize("it's a test to spot how Things happ3n héhé")
    "it-s-a-test-to-spot-how-things-happ3n-h-h"
    """

    # Vervang accenten door "-"
    s_wo_accents = unidecode.unidecode(s)
    accents = [s for s in s if s not in s_wo_accents]
    for accent in accents:
        s = s.replace(accent, "-")

    # Zet om naar kleine letters
    s = s.lower()

    # Behoud alleen alfanumerieke tekens en verwijder "-" aan het einde
    normalized = (
        "".join([char if char.isalnum() else "-" for char in s]).strip("-").lower()
    )

    return normalized
