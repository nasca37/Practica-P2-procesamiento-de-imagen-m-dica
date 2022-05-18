import numpy as np
import math
import scipy.ndimage as ndimage
from skimage.transform import resize
from scipy.optimize import least_squares


def scale_data(d,shape):

    return resize(d,shape,anti_aliasing=True)

def scale_points(list_of_points,shape_org,shape_target):
    scale = np.array(shape_target)/np.array(shape_org)
    scale = np.array([0.5,0.5,1])
    new_list = []
    for point in list_of_points:
        new_list.append(np.around(point*scale))
    return new_list

def traslacion(punto, vector_traslacion):
    x, y, z = punto
    t_1, t_2, t_3 = vector_traslacion
    punto_transformado = (x+t_1, y+t_2, z+t_3)
    return punto_transformado


def rotacion_axial(punto, angulo_en_radianes, eje_traslacion):
    x, y, z = punto
    v_1, v_2, v_3 = eje_traslacion
    #   Vamos a normalizarlo para evitar introducir restricciones en el optimizador
    v_norm = math.sqrt(sum([coord ** 2 for coord in [v_1, v_2, v_3]]))
    v_1, v_2, v_3 = v_1 / v_norm, v_2 / v_norm, v_3 / v_norm
    #   Calcula cuaternión del punto
    p = (0, x, y, z)
    #   Calcula cuaternión de la rotación
    cos, sin = math.cos(angulo_en_radianes / 2), math.sin(angulo_en_radianes / 2)
    q = (cos, sin * v_1, sin * v_2, sin * v_3)
    #   Calcula el conjugado
    q_conjugado = (cos, -sin * v_1, -sin * v_2, -sin * v_3)
    #   Calcula el cuaternión correspondiente al punto rotado
    p_prima = multiplicar_quaterniones(q, multiplicar_quaterniones(p, q_conjugado))
    # Devuelve el punto rotado
    punto_transformado = p_prima[1], p_prima[2], p_prima[3]
    return punto_transformado



def transformacion_rigida_3D(punto, parametros):
    x, y, z = punto
    t_11, t_12, t_13, alpha_in_rad, v_1, v_2, v_3  = parametros
    #   Aplicar una primera traslación
    x, y, z = traslacion(punto=(x, y, z), vector_traslacion=(t_11, t_12, t_13))
    #   Aplicar una rotación axial traslación
    x, y, z = rotacion_axial(punto=(x, y, z), angulo_en_radianes=alpha_in_rad, eje_traslacion=(v_1, v_2, v_3))
    punto_transformado = (x, y, z)
    return punto_transformado




def multiplicar_quaterniones(q1, q2):
    """Multiplica cuaterniones expresados como (1, i, j, k)."""
    return (
        q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3],
        q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2],
        q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1],
        q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]
    )


def cuaternion_conjugado(q):
    """Conjuga un cuaternión expresado como (1, i, j, k)."""
    return (
        q[0], -q[1], -q[2], -q[3]
    )


def residuos_cuadraticos(lista_puntos_ref, lista_puntos_inp):
    """Devuelve un array con los residuos cuadráticos del ajuste."""
    residuos = []
    for p1, p2 in zip(lista_puntos_ref, lista_puntos_inp):
        p1 = np.asarray(p1, dtype='float')
        p2 = np.asarray(p2, dtype='float')
        residuos.append(np.sqrt(np.sum(np.power(p1-p2, 2))))
    residuos_cuadraticos = np.power(residuos, 2)
    return residuos_cuadraticos



def funcion_a_minimizar(parametros,list_points,points_target):
    points_trans = [transformacion_rigida_3D(landmark, parametros) for landmark in list_points]
    return residuos_cuadraticos(points_target,points_trans)


def apply_traslacion_all(parametros,data,phantom_shape):
    rotated_data = np.zeros(phantom_shape)
    shape = data.shape

    print(rotated_data.shape)
    for y in range(shape[0]):
        for x in range(shape[1]):
            for z in range(shape[2]):
                punto = (x,y,z)
                trans = np.around(transformacion_rigida_3D(punto,parametros)).astype(np.uint8)
                #print(trans)
                if trans[0] >= 0 and trans[1] >= 0 and trans[2] >= 0:
                    if trans[1] < phantom_shape[0] and trans[0] < phantom_shape[1] and trans[2] < phantom_shape[2]:
                        rotated_data[trans[1],trans[0],trans[2]] = data[y,x,z]
    return rotated_data



def apply_corregistro(points_data,points_phantom,data,target):
    original_shape = data.shape
    points_scaled = scale_points(points_data,data.shape,target.shape)
    parametros = [0,0,0,0,1,0,0]



    resultado = least_squares(funcion_a_minimizar,x0=parametros,verbose=1,
                            xtol=None,
                            ftol=None,
                            bounds=([-500,-500,-500,0,0,0,0],
                                    [500,500,500,math.pi*2,1,1,1]),
                            args=(points_scaled,points_phantom), max_nfev=2000)
    para = resultado.x

    data_scaled = scale_data(data,data.shape*np.array([0.5,0.5,1]))
    print(data_scaled.shape)
    data_scaled = apply_traslacion_all(para,data_scaled,target.shape)
    error = sum(resultado.fun) / len(resultado.fun)
    print(f"El error cuadratico medio es de: {error}")
    return resultado,data_scaled


# def traslacion_complete(data,traslacion):
#     datos_trasladados = ndimage.shift(data,traslacion)
#     return datos_trasladados

# def rotacion_axial_complete(data, alpha, axis):
#     eje = math.ceil(axis)
#     ejes = [0,1,2]
#     ejes.remove(eje)

    # return ndimage.interpolation.rotate(data,alpha,ejes,reshape=False)

# def transformacion_rigida_3D(data,parametros):
#     t1, t2, t3, alpha, v = parametros

#     # traslación
#     data_traslacion = traslacion(data,(t1,t2,t3))
#     # rotación
#     data_final = rotacion_axial(data_traslacion,alpha,v)
#     return data_final


# def residuos_cuadraticos(target,data):
#     return (np.square(target-data)).mean(axis=None)

# def funcion_a_minimizar(parametros,target,data):
#     print(parametros)
#     new_data = transformacion_rigida_3D(data,parametros)
#     return residuos_cuadraticos(target,new_data)

# def apply_rotation_all(alpha,vector,data):
#     shape = data.shape
#     # rotated_data = np.zeros(shape)
#     # print(rotated_data.shape)
#     for x in range(shape[0]):
#         for y in range(shape[1]):
#             for z in range(shape[2]):
#                 punto = (x,y,z)
#                 trans = np.around(rotacion_axial(punto,alpha,vector)).astype(np.uint8)
#                 #print(trans)
#                 if trans[0] >= 0 and trans[1] >= 0 and trans[2] >= 0:
#                     if trans[0] < shape[0] and trans[1] < shape[1] and trans[2] < shape[2]:
#                         data[trans[0],trans[1],trans[2]] = data[x,y,z]
#     return data


# def apply_corregistro(points_data,points_phantom,data):
#     original_shape = data.shape
#     data_scaled = scale_data(data,target)
#     parametros = [0,0,0,0,1,0,0]
#     resultado = least_squares(funcion_a_minimizar,x0=parametros,verbose=1,
#                             bounds=([-target.shape[0],-target.shape[1],-target.shape[2],0,0,0,0],
#                                     [target.shape[0],target.shape[1],target.shape[2],math.pi*2,1,1,1]),
#                             args=(target,data_scaled))
#     para = resultado.x
#     data_scaled = traslacion(data_scaled,(para[0],para[1],para[2]))
#     data_scaled = rotacion_axial(data_scaled,para[3],para[4])
#     #data = np.resize(data_scaled,original_shape)
#     data = resize(data_scaled,original_shape,anti_aliasing=True)
#     print(data)
#     return resultado,data