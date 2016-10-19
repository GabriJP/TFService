def getclassnum():
    # return int(input("Número de clases a usar: "))
    return 1


def getclassname(num):
    # return input("Nombre de la clase %d: " % num)
    return "1"


def getclassdirectory(num):
    # return input("Directorio de la clase %d: " % num)
    return "Other/Classes/1"


def getredimension():
    # raw = input("Redimensionar a: ")
    # height = raw[0:raw.index('x')]
    # width = raw[raw.index('x')+1:]
    # return [height, width]
    return 500, 500


def getcrop():
    # raw = input("Recortar a: ")
    # height = raw[0:raw.index('x')]
    # width = raw[raw.index('x') + 1:]
    # raw = input("Empezando en: ")
    # x = raw[0:raw.index('x')]
    # y = raw[raw.index('x') + 1:]
    # return [height, width, x, y]
    return 100, 100, 400, 400


def getpercentages():
    # return int(input("Porcentage para entrenamiento: "))
    return 80


def getoutputdrectory():
    # return input("Directorio de salida: ")
    return "output"
