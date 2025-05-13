import os
import cv2  # Para la cámara web
from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
from clarifai_grpc.grpc.api.status import status_code_pb2

# --- CONFIGURACIÓN DE CLARIFAI ---
CLARIFAI_PAT = "ASI MERO"  # Tu Personal Access Token

# MODEL_URL que proporcionaste. El script intentará limpiarla.
# Asegúrate de que la URL base (antes del '?') apunte a un modelo válido y activo.
MODEL_URL = "https://clarifai.com/clarifai/main/models/face-detection?inputId=https%3A%2F%2Fs3.amazonaws.com%2Fsamples.clarifai.com%2Ffeatured-models%2Fface-family-with-light-blue-shirts.jpg"


def capturar_y_guardar_foto(nombre_archivo="captura_cf.jpg"):
    """Captura una foto desde la webcam y la guarda en un archivo."""
    cap = cv2.VideoCapture(0)  # 0 es usualmente la cámara por defecto

    if not cap.isOpened():
        print("Error: No se pudo abrir la cámara.")
        return None

    print("Cámara abierta. Presiona 'espacio' para tomar la foto, o 'esc' para salir.")

    ret, frame = cap.read()  # Intenta leer un frame inicial
    if not ret:
        print("Error: No se pudo capturar un frame inicial de la cámara.")
        cap.release()
        cv2.destroyAllWindows()
        return None

    cv2.imshow("Presiona ESPACIO para capturar", frame)

    foto_capturada = False
    while True:
        ret, frame = cap.read()  # Lee continuamente para mantener la ventana activa
        if not ret:
            print("Error al leer frame de la cámara.")
            break
        cv2.imshow("Presiona ESPACIO para capturar", frame)

        k = cv2.waitKey(1) & 0xFF  # '& 0xFF' es buena práctica en todas las plataformas
        if k == 27:  # ESC presionado
            print("Cerrando cámara sin capturar.")
            nombre_archivo = None  # No se guardó imagen
            break
        elif k == 32:  # ESPACIO presionado
            try:
                cv2.imwrite(nombre_archivo, frame)
                print(f"¡Foto guardada como {nombre_archivo}!")
                foto_capturada = True
            except Exception as e:
                print(f"Error al guardar la imagen: {e}")
                nombre_archivo = None
            break

    cap.release()
    cv2.destroyAllWindows()
    return nombre_archivo if foto_capturada else None


def analizar_imagen_clarifai(ruta_imagen):
    """Envía la imagen a Clarifai y detecta caras/personas."""
    if not ruta_imagen:
        print("Error: No se proporcionó ruta de imagen para analizar.")
        return "Error: No se proporcionó ruta de imagen."

    print(f"Analizando imagen local con Clarifai: {ruta_imagen}")

    try:
        if CLARIFAI_PAT == "TU_PAT_DE_CLARIFAI_AQUI":  # Chequeo por si acaso
            return "Error: El PAT de Clarifai no ha sido configurado en el script."

        channel = ClarifaiChannel.get_grpc_channel()
        stub = service_pb2_grpc.V2Stub(channel)

        metadata = (('authorization', f'Key {CLARIFAI_PAT}'),)

        with open(ruta_imagen, "rb") as f:
            file_bytes = f.read()

        # --- MODIFICACIÓN AQUÍ para limpiar la MODEL_URL ---
        try:
            base_model_url = MODEL_URL.split('?')[0]  # Tomamos la parte antes del '?'
            parts = base_model_url.strip("/").split("/")

            # Asegurarnos que tenemos suficientes partes después de limpiar la URL
            if len(parts) < 4:  # Necesitamos al menos scheme, domain, user, app, "models", model_id
                raise IndexError("URL no tiene suficientes segmentos para extraer user_id, app_id, model_id.")

            model_user_id = parts[-4]  # e.g., 'clarifai'
            model_app_id = parts[-3]  # e.g., 'main'
            model_id_from_url = parts[-1]  # e.g., 'face-detection'

            print(f"Usando User ID: {model_user_id}, App ID: {model_app_id}, Model ID: {model_id_from_url}")

        except IndexError as e:
            print(f"Error procesando MODEL_URL ('{MODEL_URL}'): {e}")
            return "Error: El formato de MODEL_URL no es válido o no se pudieron extraer los componentes. Debe ser como 'https://clarifai.com/USER_ID/APP_ID/models/MODEL_ID'"
        # --- FIN DE LA MODIFICACIÓN ---

        post_model_outputs_response = stub.PostModelOutputs(
            service_pb2.PostModelOutputsRequest(
                user_app_id=resources_pb2.UserAppIDSet(user_id=model_user_id, app_id=model_app_id),
                model_id=model_id_from_url,
                inputs=[
                    resources_pb2.Input(
                        data=resources_pb2.Data(
                            image=resources_pb2.Image(
                                base64=file_bytes
                            )
                        )
                    )
                ]
            ),
            metadata=metadata
        )

        if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
            print(
                f"Error en la API de Clarifai: {post_model_outputs_response.status.description} (Código: {post_model_outputs_response.status.code})")
            return f"Error en la predicción de Clarifai: {post_model_outputs_response.status.description}"

        print("\n--- Resultados del Análisis (Clarifai) ---")
        persona_detectada = False
        conceptos_encontrados_cf = []
        objetos_descritos = []

        if post_model_outputs_response.outputs:
            for output in post_model_outputs_response.outputs:
                if output.data and output.data.regions:
                    print(f"Regiones detectadas: {len(output.data.regions)}")
                    for region_idx, region in enumerate(output.data.regions):
                        region_conceptos = []
                        if region.data and region.data.concepts:
                            for concept in region.data.concepts:
                                concepto_str = f"{concept.name} (Confianza: {concept.value:.2f})"
                                region_conceptos.append(concepto_str)
                                if "face" in concept.name.lower() or "person" in concept.name.lower():
                                    persona_detectada = True
                        if region_conceptos:
                            objetos_descritos.append(f"Región {region_idx + 1}: {', '.join(region_conceptos)}")
                            conceptos_encontrados_cf.extend(region_conceptos)
                elif output.data and output.data.concepts:
                    print("Conceptos generales detectados:")
                    for concept in output.data.concepts:
                        concepto_str = f"{concept.name} (Confianza: {concept.value:.2f})"
                        objetos_descritos.append(f"General: {concepto_str}")
                        conceptos_encontrados_cf.append(concepto_str)
                        print(f"- {concepto_str}")
                        if "person" in concept.name.lower() or "people" in concept.name.lower() or "face" in concept.name.lower():
                            persona_detectada = True

        resultado_final_str = ""
        if persona_detectada:
            resultado_final_str = "¡PERSONA DETECTADA (Clarifai)!\n"
        else:
            resultado_final_str = "No se detectó ninguna persona (Clarifai).\n"

        if objetos_descritos:
            resultado_final_str += "Objetos/Conceptos identificados:\n" + "\n".join(
                [f"  - {desc}" for desc in objetos_descritos])
        elif conceptos_encontrados_cf:
            resultado_final_str += "Conceptos generales identificados:\n" + "\n".join(
                [f"  - {desc}" for desc in conceptos_encontrados_cf])

        return resultado_final_str if resultado_final_str else "No se obtuvieron resultados claros del análisis."

    except Exception as e:
        return f"Error inesperado al analizar con Clarifai: {str(e)}"


def main():
    print("Iniciando Reto IA con Clarifai")

    nombre_foto_cf = "foto_reto_clarifai.jpg"

    # Ya no se necesita la condición de ejemplo para MODEL_URL aquí,
    # porque la URL que tienes ahora es la que estás probando.
    # El PAT sí es importante que no sea el placeholder.
    if CLARIFAI_PAT == "TU_PAT_DE_CLARIFAI_AQUI" or not MODEL_URL:
        print("-" * 50)
        print("¡ATENCIÓN! CONFIGURACIÓN REQUERIDA:")
        print("1. Debes configurar tu 'CLARIFAI_PAT' (Personal Access Token) en el script.")
        print("2. Debes asegurarte que 'MODEL_URL' esté definida y apunte a un modelo válido de Clarifai.")
        print("-" * 50)
        return

    ruta_foto_capturada = capturar_y_guardar_foto(nombre_foto_cf)

    if ruta_foto_capturada:
        resultado_analisis = analizar_imagen_clarifai(
            ruta_imagen=ruta_foto_capturada)  # Asegúrate de pasar el argumento correcto
        print("\n========= RESULTADO FINAL (Clarifai) ==========")
        print(resultado_analisis)
        print("==============================================")
    else:
        print("No se tomó ninguna foto, o se canceló la captura.")


if __name__ == "__main__":
    main()