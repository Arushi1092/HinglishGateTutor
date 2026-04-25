import wikipediaapi

def fetch_wikipedia_content(title: str, lang: str = 'en') -> str:
    user_agent = "HindiGateTutor/1.0 (https://github.com/yourusername/HindiGateTutor; your@email.com)"
    wiki = wikipediaapi.Wikipedia(user_agent=user_agent, language=lang)
    page = wiki.page(title)
    
    if not page.exists():
        return None
    
    return page.text
