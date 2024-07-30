import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import ImageEmbedder, ImageEmbedderOptions, RunningMode
from colorama import init, Fore, Style

init(autoreset=True)

ROOT_FOLDER = "/Users/alisson/Downloads/ia/"

MODELO = ROOT_FOLDER + "modelos/mobilenet_v3_small_075_224_embedder.tflite"

GATOS_BRANCOS = [
    ROOT_FOLDER + "imagens/imagens/gato_branco1.jpg",
    ROOT_FOLDER + "imagens/imagens/gato_branco2.jpg"
]

GATOS_PRETOS = [
    ROOT_FOLDER + "imagens/imagens/gato_preto1.jpg",
    ROOT_FOLDER + "imagens/imagens/gato_preto2.jpg",
]

CACHORROS = [
    ROOT_FOLDER + "imagens/imagens/cachorro1.jpg",
    ROOT_FOLDER + "imagens/imagens/cachorro2.jpg",
]

BICHOS_PARA_TESTES = [
    {
        "tipo": "cachorro",
        "imagem": ROOT_FOLDER + "embeddings/imagens/cachorro_teste.jpg",
    },
    {
        "tipo": "gato_branco",
        "imagem": ROOT_FOLDER + "embeddings/imagens/gato_branco_teste.jpg",
    },
    {
        "tipo": "gato_preto",
        "imagem": ROOT_FOLDER + "embeddings/imagens/gato_preto_teste.jpg",
    },
]

def configurar():
    try:
        opcoes = ImageEmbedderOptions(
            base_options=BaseOptions(model_asset_path=MODELO),
            quantize=True,
            running_mode=RunningMode.IMAGE
        )
        incorporador = ImageEmbedder.create_from_options(opcoes)
        return True, incorporador
    except Exception as e:
        print("Erro ao configurar o incorporador: " + str(e))
        return False, None

def processar(imagem, incorporador):
    try:
        imagem = mp.Image.create_from_file(imagem)
        incorporacao = incorporador.embed(imagem)
        return True, incorporacao
    except Exception as e:
        print("Erro ao processar a imagem: " + str(e))
        return False, None

def processar_bichos(imagens, incorporador):
    incorporacoes = []
    for imagem in imagens:
        processada, incorporacao = processar(imagem, incorporador)
        if processada:
            incorporacoes.append(incorporacao)
    return incorporacoes

def comparar(imagem, grupo_candidato, incorporador):
    processada, incorporacao = processar(imagem, incorporador)
    if not processada:
        return 0.0
    
    similaridade = max(
        incorporador.cosine_similarity(incorporacao.embeddings[0], individuo.embeddings[0])
        for individuo in grupo_candidato
    )
    return similaridade

def testar_similaridade(bicho, gatos_brancos, gatos_pretos, cachorros, incorporador):
    similaridade_gatos_brancos = comparar(bicho['imagem'], gatos_brancos, incorporador)
    similaridade_gatos_pretos = comparar(bicho['imagem'], gatos_pretos, incorporador)
    similaridade_cachorros = comparar(bicho['imagem'], cachorros, incorporador)

    print(f"{Fore.YELLOW}Testando similaridade entre {bicho['tipo']} e gatos brancos")
    print(f"{Fore.GREEN}Similaridade: {similaridade_gatos_brancos}")

    print(f"{Fore.YELLOW}Testando similaridade entre {bicho['tipo']} e gatos pretos")
    print(f"{Fore.GREEN}Similaridade: {similaridade_gatos_pretos}")

    print(f"{Fore.YELLOW}Testando similaridade entre {bicho['tipo']} e cachorros")
    print(f"{Fore.GREEN}Similaridade: {similaridade_cachorros}")

    if similaridade_cachorros > max(similaridade_gatos_brancos, similaridade_gatos_pretos):
        print(f"{Fore.CYAN}Tem maior similaridade com cachorro")
    elif similaridade_gatos_pretos > similaridade_gatos_brancos:
        print(f"{Fore.CYAN}Tem maior similaridade com gatos pretos")
    else:
        print(f"{Fore.CYAN}Tem maior similaridade com gatos brancos")

if __name__ == "__main__":
    configurado, incorporador = configurar()

    if configurado:
        gatos_pretos = processar_bichos(GATOS_PRETOS, incorporador)
        gatos_brancos = processar_bichos(GATOS_BRANCOS, incorporador)
        cachorros = processar_bichos(CACHORROS, incorporador)

        for bicho in BICHOS_PARA_TESTES:
            testar_similaridade(bicho, gatos_brancos, gatos_pretos, cachorros, incorporador)
