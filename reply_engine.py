import re

ASPECT_KEYWORDS = {
    "Delivery": ["delivery", "delayed", "late", "arrived", "shipping", "courier", "package"],
    "Food Quality": ["food", "taste", "delicious", "bland", "cold", "meal", "flavor", "quality"],
    "Customer Service": ["staff", "service", "waiter", "support", "rude", "friendly", "helpful"],
    "Pricing": ["price", "expensive", "overpriced", "cost", "bill", "cheap", "worth"]
}

def extract_aspect(review_text: str) -> str:
    """Identifies the primary topic/aspect mentioned in the review."""
    text_lower = review_text.lower()
    for aspect, keywords in ASPECT_KEYWORDS.items():
        if any(re.search(rf"\b{k}\b", text_lower) for k in keywords):
            return aspect
    return "Overall Experience"

def generate_reply(sentiment: str, aspect: str, customer_name: str = "Valued Customer") -> str:
    """Generates dynamic tailored responses based on sentiment and aspect."""
    if sentiment == "Positive":
        return (
            f"Dear {customer_name},\n\n"
            f"Thank you so much for your wonderful review! We're thrilled to hear that you had a great experience "
            f"with our {aspect.lower()}. Our team works hard to keep standards high, and your support means the world to us.\n\n"
            f"We look forward to welcoming you back soon!\n— The Management Team"
        )
    elif sentiment == "Neutral":
        return (
            f"Dear {customer_name},\n\n"
            f"Thank you for taking the time to share your feedback regarding our {aspect.lower()}. We appreciate your balanced "
            f"perspective and are always looking for ways to improve our offerings.\n\n"
            f"If you have specific suggestions on how we can make your next visit a 5-star experience, please let us know!\n— The Management Team"
        )
    else:  #for negative sentiment
        return (
            f"Dear {customer_name},\n\n"
            f"We sincerely apologize for your disappointing experience with our {aspect.lower()}. This is certainly not "
            f"the standard of quality and service we aim to deliver.\n\n"
            f"We would appreciate the opportunity to make this right for you. Please reach out to our management team directly "
            f"at support@example.com so we can investigate and resolve this issue.\n— Customer Care Team"
        )