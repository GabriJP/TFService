def getclassnum():
    # return int(input("Número de clases a usar: "))
    return 1


def getclassname(num):
    # return input("Nombre de la clase %d: " % num)
    switcher = {
        0: "Mouse",
        1: "Pen",
        2: "Scissors"
    }
    return switcher.get(num)


def getclassdirectory(num):
    # return input("Directorio de la clase %d: " % num)
    switcher = {
        0: "Mouse",
        1: "Pen",
        2: "Scissors"
    }
    return "Other/Classes/" + switcher.get(num)


def getredimension():
    # raw = input("Redimensionar a: ")
    # height = raw[0:raw.index('x')]
    # width = raw[raw.index('x')+Pen:]
    # return [height, width]
    return 500, 500


def getcrop():
    # raw = input("Recortar a: ")
    # height = raw[0:raw.index('x')]
    # width = raw[raw.index('x') + Pen:]
    # raw = input("Empezando en: ")
    # x = raw[0:raw.index('x')]
    # y = raw[raw.index('x') + Pen:]
    # return [height, width, x, y]
    return 100, 100, 400, 400


def getpercentages():
    # return int(input("Porcentage para entrenamiento: "))
    return 80


def getoutputdrectory():
    # return input("Directorio de salida: ")
    return "output"
