from transformers import pipeline

# Creamos un pipeline de clasificación de sentimientos usando el modelo especificado
clasificador = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def clasificar_sentimiento(texto):
    """
    Clasifica el sentimiento de un texto como positivo o negativo.

    Args:
        texto (str): El texto a analizar.

    Returns:
        dict: Un diccionario con la etiqueta ('POSITIVE' o 'NEGATIVE') y la puntuación.
    """
    resultado = clasificador(texto)[0]
    return {"sentimiento": resultado['label'], "confianza": resultado['score']}

# Ejemplo de uso
if __name__ == "__main__":
    texto_ejemplo = "Odio el café"
    resultado = clasificar_sentimiento(texto_ejemplo)
    print(f"Texto: {texto_ejemplo}")
    print(f"Sentimiento: {resultado['sentimiento']} (confianza: {resultado['confianza']:.2f})")
