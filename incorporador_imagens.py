import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import ImageEmbedder, ImageEmbedderOptions, RunningMode

from colorama import Back, Style

MODELO = "/Users/alisson/Downloads/ia/modelos/mobilenet_v3_small_075_224_embedder.tflite"

GATOS_BRANCOS = [
    "/Users/alisson/Downloads/ia/imagens/imagens/gato_branco1.jpg",
    "/Users/alisson/Downloads/ia/imagens/imagens/gato_branco2.jpg"
]

GATOS_PRETOS = [
    "/Users/alisson/Downloads/ia/imagens/imagens/gato_preto1.jpg",
    "/Users/alisson/Downloads/ia/imagens/imagens/gato_preto2.jpg",
]

CACHORROS = [
    "/Users/alisson/Downloads/ia/imagens/imagens/cachorro1.jpg",
    "/Users/alisson/Downloads/ia/imagens/imagens/cachorro2.jpg",
]

BICHOS_PARA_TESTES = [
    {
        "tipo": "cachorro",
        "imagem": "/Users/alisson/Downloads/ia/embeddings/imagens/cachorro_teste.jpg",
    },
    {
        "tipo": "gato_branco",
        "imagem": "/Users/alisson/Downloads/ia/embeddings/imagens/gato_branco_teste.jpg",
    },
    {
        "tipo": "gato_preto",
        "imagem": "/Users/alisson/Downloads/ia/embeddings/imagens/gato_preto_teste.jpg",
    },
]

def configurar():
    configurado, incorporador = False, None

    try:
        opcoes = ImageEmbedderOptions(
            base_options=BaseOptions(
                model_asset_path=MODELO
                ), quantize=True, running_mode=RunningMode.IMAGE)
        incorporador = ImageEmbedder.create_from_options(opcoes)
        configurado = True
    except Exception as e:
        print("erroooo " + str(e))

    return configurado, incorporador


def processar(imagem, incorporador):
    processada, incorporacao = False, None

    try:
        imagem = mp.Image.create_from_file(imagem)
        incorporacao = incorporador.embed(imagem)
        processada = True
    except Exception as e:
        print("erroooo " + str(e))

    return processada, incorporacao

def processar_bichos(imagens, incorporador):
    processados, incorporacoes = False, []

    for imagem in imagens:
        processada, incorporacao = processar(imagem, incorporador)
        if processada:
            incorporacoes.append(incorporacao)

    return processados, incorporacoes

def comparar(imagem, grupo_candidato, incorporador):
    similaridade = 0.0

    processada, incorporacao = processar(imagem, incorporador)
    if processada:
        for individuo in grupo_candidato[1]:
            nova_similaridade = incorporador.cosine_similarity(incorporacao.embeddings[0], individuo.embeddings[0])
            
            similaridade = nova_similaridade if nova_similaridade > similaridade else similaridade

    return similaridade

if __name__ == "__main__":
    configurado, incorporador = configurar()

    if configurado:
        gatos_pretos = processar_bichos(GATOS_PRETOS, incorporador)
        gatos_brancos = processar_bichos(GATOS_BRANCOS, incorporador)
        cachorros = processar_bichos(CACHORROS, incorporador)

        for bicho in BICHOS_PARA_TESTES:
            print(f"testando similaridade entre {bicho['tipo']} e gatos brancos")
            similaridade_com_gatos_brancos = comparar(bicho['imagem'], gatos_brancos, incorporador)

            print(f"testando similaridade entre {bicho['tipo']} e gatos pretos")
            similaridade_com_gatos_pretos = comparar(bicho['imagem'], gatos_pretos, incorporador)

            print(f"testando similaridade entre {bicho['tipo']} e cachorros")
            similaridade_com_cachorros = comparar(bicho['imagem'], cachorros, incorporador)

            if similaridade_com_cachorros > similaridade_com_gatos_brancos and similaridade_com_cachorros > similaridade_com_gatos_pretos:
                print(f"tem maior similaridade com cachorro")
            
            if similaridade_com_gatos_pretos > similaridade_com_gatos_brancos and similaridade_com_gatos_pretos > similaridade_com_cachorros:
                print("TEM MAIOR SIMILARIDADE COM GATOS PRETOS")

            if similaridade_com_gatos_brancos > similaridade_com_gatos_pretos and similaridade_com_gatos_brancos > similaridade_com_cachorros:
                print("TEM maior similaridade com gatos brancos")