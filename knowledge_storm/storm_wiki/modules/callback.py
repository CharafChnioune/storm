class BaseCallbackHandler:
    """
    Basis callback handler die gebruikt kan worden om callbacks af te handelen van de STORM pipeline.
    Deze klasse definieert de interface voor verschillende stadia in het proces van artikelgeneratie.
    Subklassen kunnen deze methoden overschrijven om aangepast gedrag te implementeren.
    """

    def on_identify_perspective_start(self, **kwargs):
        """
        Wordt uitgevoerd wanneer de perspectiefidentificatie start.
        
        Args:
            **kwargs: Variabele keyword argumenten voor toekomstige uitbreidbaarheid.
        """
        pass

    def on_identify_perspective_end(self, perspectives: list[str], **kwargs):
        """
        Wordt uitgevoerd wanneer de perspectiefidentificatie is voltooid.
        
        Args:
            perspectives (list[str]): Lijst van geïdentificeerde perspectieven.
            **kwargs: Variabele keyword argumenten voor toekomstige uitbreidbaarheid.
        """
        pass

    def on_information_gathering_start(self, **kwargs):
        """
        Wordt uitgevoerd wanneer het verzamelen van informatie start.
        
        Args:
            **kwargs: Variabele keyword argumenten voor toekomstige uitbreidbaarheid.
        """
        pass

    def on_dialogue_turn_end(self, dlg_turn, **kwargs):
        """
        Wordt uitgevoerd wanneer een vraag-antwoord ronde is voltooid.
        
        Args:
            dlg_turn: Informatie over de voltooide dialoogronde.
            **kwargs: Variabele keyword argumenten voor toekomstige uitbreidbaarheid.
        """
        pass

    def on_information_gathering_end(self, **kwargs):
        """
        Wordt uitgevoerd wanneer het verzamelen van informatie is voltooid.
        
        Args:
            **kwargs: Variabele keyword argumenten voor toekomstige uitbreidbaarheid.
        """
        pass

    def on_information_organization_start(self, **kwargs):
        """
        Wordt uitgevoerd wanneer de organisatie van informatie start.
        
        Args:
            **kwargs: Variabele keyword argumenten voor toekomstige uitbreidbaarheid.
        """
        pass

    def on_direct_outline_generation_end(self, outline: str, **kwargs):
        """
        Wordt uitgevoerd wanneer de directe outlinegeneratie is voltooid.
        
        Args:
            outline (str): De gegenereerde outline.
            **kwargs: Variabele keyword argumenten voor toekomstige uitbreidbaarheid.
        """
        pass

    def on_outline_refinement_end(self, outline: str, **kwargs):
        """
        Wordt uitgevoerd wanneer de verfijning van de outline is voltooid.
        
        Args:
            outline (str): De verfijnde outline.
            **kwargs: Variabele keyword argumenten voor toekomstige uitbreidbaarheid.
        """
        pass
