import sys
import os
import struct
import numpy as np
import tifffile as tf


def reparar_video_tiff(ruta_origen, ruta_destino):
    """Repara un TIFF de ImageJ con la cadena de IFDs rota y/o truncado.

    ImageJ guarda los stacks como un único IFD seguido de todos los frames
    de forma CONTIGUA (sin IFDs intermedios). Si el puntero al siguiente IFD
    se corrompe, tifffile solo expone 1 pagina aunque haya miles de frames.
    Aca leemos el primer IFD a mano y extraemos el bloque contiguo completo.
    """
    print(f"Inspeccionando: {ruta_origen}...")

    with open(ruta_origen, "rb") as f:
        head = f.read(8)
        bo = ">" if head[:2] == b"MM" else "<"
        ifd0_off = struct.unpack(bo + "I", head[4:8])[0]

        f.seek(ifd0_off)
        n = struct.unpack(bo + "H", f.read(2))[0]
        tags = {}
        for _ in range(n):
            tag, typ, cnt = struct.unpack(bo + "HHI", f.read(8))
            tags[tag] = (typ, f.read(4))

        def value(tag):
            # Respeta el tipo: 3=SHORT (2 bytes), 4=LONG (4 bytes).
            typ, raw = tags[tag]
            if typ == 3:
                return struct.unpack(bo + "H", raw[:2])[0]
            return struct.unpack(bo + "I", raw)[0]

        width = value(256)           # ImageWidth
        height = value(257)          # ImageLength
        bits = value(258)            # BitsPerSample
        data_off = value(273)        # StripOffsets (inicio del frame 0)
        frame_bytes = value(279)     # StripByteCounts (bytes por frame)

        if bits != 8:
            raise ValueError(f"Solo soportado uint8 por ahora (bits={bits}).")

        # Cuantos frames dice la metadata de ImageJ (tag 270 = ImageDescription)
        declarados = None
        if 270 in tags:
            desc_off = struct.unpack(bo + "I", tags[270][1])[0]
            f.seek(desc_off)
            desc = f.read(512).split(b"\x00")[0].decode("latin1", "ignore")
            for line in desc.splitlines():
                if line.startswith("images="):
                    declarados = int(line.split("=")[1])

    filesize = os.path.getsize(ruta_origen)
    disponibles = (filesize - data_off) // frame_bytes

    print(f"  Frame: {width}x{height} uint8 ({frame_bytes} bytes/frame)")
    if declarados is not None:
        print(f"  Frames declarados por ImageJ: {declarados}")
    print(f"  Frames completos presentes en el archivo: {disponibles}")
    if declarados is not None and disponibles < declarados:
        print(f"  [AVISO] Archivo TRUNCADO: faltan {declarados - disponibles} frames.")
        print(f"          Se recuperan los {disponibles} frames intactos.")

    if disponibles < 1:
        raise ValueError("No hay ni un frame completo en el archivo.")

    # Lectura contigua y eficiente via memmap (no carga todo en RAM de golpe)
    data = np.memmap(
        ruta_origen,
        dtype=np.uint8,
        mode="r",
        offset=data_off,
        shape=(disponibles, height, width),
    )

    print(f"Guardando {disponibles} frames en: {ruta_destino}...")
    tf.imwrite(ruta_destino, data, imagej=True)
    print("Listo.")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("\n[Uso] python fix_tif.py archivo_roto.tif archivo_arreglado.tif\n")
        sys.exit(1)
    reparar_video_tiff(sys.argv[1], sys.argv[2])
