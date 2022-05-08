
from audioop import reverse
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from pydicom.fileset import FileSet
import PySimpleGUI as sg
import glob,os
import cv2
from skimage import measure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import ndimage
from pydicom.pixel_data_handlers.util import apply_modality_lut
import random


# Variables globales
SLICE_X = 0
SLICE_Y = 0
SLICE_Z = 0

MAX_SLIDER_X = 0
MAX_SLIDER_Y = 0
MAX_SLIDER_Z = 0

SLICE_X_2 = 0
SLICE_Y_2 = 0
SLICE_Z_2 = 0

ZONAS_ATLAS = 170
 
GENERAL_ATLAS = False
ONLY_HIPO = False

MAX_SLIDER_X_2 = 0
MAX_SLIDER_Y_2 = 0
MAX_SLIDER_Z_2 = 0

HIPO_L = 121
HIPO_R = 150

SUBIMAGE = False
SUB_POINT_1 = None
SUB_POINT_2 = None 
SECOND_SIZE = (4,8)
AXIS = 2
THRESHOLD = 0.2

SEGMENTATION = False
MASK = None

INTERCEPT = 0
SLOPE = 0


_VARS = {'fig1': False,
         'fig2': False,
         'fig3': False,
         'fig4': False,
         'fig5': False,
         'fig6': False,
         'fig_second':False,
         'pltFig': False}

def draw_figure(canvas, figure):
    figure_canvas_agg = FigureCanvasTkAgg(figure, canvas)
    figure_canvas_agg.draw()
    figure_canvas_agg.get_tk_widget().pack(side='bottom', fill='none', expand=0)
    return figure_canvas_agg

def from_hu_to_ct(value):
    return (value - INTERCEPT) / SLOPE

def load_dicom_folder(path):
    global SLOPE,INTERCEPT
    if path == None:
        return None
    file_paths = glob.glob(path+"/*.dcm")

    img_data = None
    idx = 0
    slices = []
    for f in file_paths:
        dcm = pydicom.dcmread(f)
        slices.append(dcm)
    slices = sorted(slices,key=lambda s: s.SliceLocation,reverse=True)
    for s in slices:
        
        
        img = s.pixel_array
        if idx != 0:
            img_data[:,:,idx] = img
        else:
            img_data = np.zeros((img.shape[0],img.shape[1],len(file_paths)))
            img_data[:,:,idx] = img
        idx = idx + 1
    return np.rot90(img_data,k=2),slices

def random_color_list(value):
    colores = []
    for i in range(value):
        r = random.randint(0,255)
        g = random.randint(0,255)
        b = random.randint(0,255)
        colores.append([r,g,b])
    return colores


def load_dicom_file(path,atlas = False):
    if path == None:
        return None

    dcm = pydicom.dcmread(path)
    if atlas:
        return dcm.pixel_array

    return dcm.pixel_array[6:-6, 6:-6, 6:-6]

def obtain_image_slice(data,lower=False):
    #global SLICE_X,SLICE_Y,SLICE_Z
    if lower:
        return (data[:,SLICE_Y2-1,:] ,data[:,:,SLICE_Z2-1],data[SLICE_X2-1,:,:])
    else:
        return (data[SLICE_X-1,:,:],data[:,SLICE_Y-1,:] ,data[:,:,SLICE_Z-1])


def subimage():
    global current_image
    global SUB_POINT_1
    global SUB_POINT_2
    global SUBIMAGE
    if SUB_POINT_1[0] < SUB_POINT_2[0] and SUB_POINT_1[1] < SUB_POINT_2[1]:
        current_image = current_image[SUB_POINT_1[1]:SUB_POINT_2[1],SUB_POINT_1[0]:SUB_POINT_2[0]]
        print(f"Punto 1: {SUB_POINT_1}   Punto 2: {SUB_POINT_2}        1")
    elif SUB_POINT_1[0] > SUB_POINT_2[0] and SUB_POINT_1[1] < SUB_POINT_2[1]:
        current_image = current_image[SUB_POINT_1[1]:SUB_POINT_2[1],SUB_POINT_2[0]:SUB_POINT_1[0]]
        print(f"Punto 1: {SUB_POINT_1}   Punto 2: {SUB_POINT_2}        2")
    elif SUB_POINT_1[0] < SUB_POINT_2[0] and SUB_POINT_1[1] > SUB_POINT_2[1]:
        print(f"Punto 1: {SUB_POINT_1}   Punto 2: {SUB_POINT_2}        3")
        current_image = current_image[SUB_POINT_2[1]:SUB_POINT_1[1],SUB_POINT_1[0]:SUB_POINT_2[0]]
    elif SUB_POINT_1[0] > SUB_POINT_2[0] and SUB_POINT_1[1] > SUB_POINT_2[1]:
        print(f"Punto 1: {SUB_POINT_1}   Punto 2: {SUB_POINT_2}        4")
        current_image = current_image[SUB_POINT_2[1]:SUB_POINT_1[1],SUB_POINT_2[0]:SUB_POINT_1[0]]

    clean_canvas('fig_second')
    show_canvas_sec(current_image)
    SUBIMAGE = False
    SUB_POINT_1, SUB_POINT_2 = None,None

def get_aspect(axis):
    if axis >= 2:
        return 1
    elif axis >= 3:
        return 1/3
    else:
        return 3

def obtain_rgb_mask(m):
    mask = np.zeros((m.shape[0],m.shape[1],3))

    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i][j] == True:
                mask[i][j] = [0,255,0]

    return mask

def obtain_rgb_mask_tones(m,colores):
    mask = np.zeros((m.shape[0],m.shape[1],3))
    

    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            if m[i][j] > 0:
                
                mask[i][j] = colores[m[i,j]]

    return mask

def show_canvas(imgs,lower = False):
    global current_image,MASK

    if lower:
        offset = 3
    else:
        offset = 0

    colores = random_color_list(255)
    for i in range(len(imgs)):
        canvas = "fig"+ str(i+1+offset)
        clean_canvas(canvas)
        fig = plt.figure(figsize=(4,4))
        plt.axis("off")
        plt.tight_layout(pad=0)
        img = imgs[i]
        idx = i + offset
        if idx < 2:
            img = np.rot90(img,k=3)
            print(img.shape)
        elif idx == 3:
            img = np.rot90(img,k=2)
        elif idx == 4:
            img = np.rot90(img,k=2)

        if GENERAL_ATLAS:
            if idx > 2:
                mask = obtain_image_slice(atlas_data,lower=True)[i]
                if idx == 3 or idx == 4:
                    mask = np.rot90(mask,k=2)
                mask = obtain_rgb_mask_tones(mask,colores)

                img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
                img = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_GRAY2RGB)
                img = algoritmo_pintor(img,mask,0.4)
                img = img.astype(np.uint8)
        if ONLY_HIPO:
              if idx > 2:
                atlas_mask = mask = np.ma.masked_inside(atlas_data,HIPO_L,HIPO_R).mask*1
                mask = obtain_image_slice(atlas_mask,lower=True)[i]
                
                if idx == 3 or idx == 4:
                    mask = np.rot90(mask,k=2)

                
                print(mask)
                mask = obtain_rgb_mask_tones(mask,colores)

                img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
                img = cv2.cvtColor(img.astype(np.uint8),cv2.COLOR_GRAY2RGB)
                img = algoritmo_pintor(img,mask,0.4)
                img = img.astype(np.uint8)          

        plt.imshow(img,cmap=plt.cm.get_cmap("bone"),aspect=get_aspect(i+offset))
        _VARS[canvas] = draw_figure(window[canvas].TKCanvas, fig)

def show_canvas_sec(img):
    clean_canvas('fig_second')
    fig = plt.figure()
    plt.imshow(img,cmap=plt.cm.get_cmap("bone"),aspect=get_aspect())
    _VARS['fig_second'] = draw_figure(window['-CANVAS2-'].TKCanvas, fig)

def clean_canvas(key):
    if _VARS[key] != False:
        _VARS[key].get_tk_widget().forget()
    
def apply_windowing(min,max):
    global current_image

    img_max = np.max(current_image)
    img_min = np.min(current_image)
    new_min = from_hu_to_ct(min)
    new_max = from_hu_to_ct(max) 
    
    r = (img_max - img_min) / (new_max-new_min+2) # unit of stretching
    out = np.round(r*(current_image-new_min+1)).astype(current_image.dtype) # stretched values
    
    out[current_image<new_min] = img_min
    out[current_image>new_max] = img_max
    
    current_image = out
    return out

def isocontorno(data,x,y,z):
    threshold = 0.1
    mask = np.zeros(data.shape)
    valor = data[y,x,z]
    print(f"En las coordenadas {x} {y} {z} tenemos el valor {valor}")

    mask = np.ma.masked_inside(data,valor-valor*threshold,np.max(data)).mask*1
    struct = ndimage.generate_binary_structure(3, 1)
    struct2 = ndimage.generate_binary_structure(3, 3)
    erodedMask = ndimage.binary_erosion(mask, structure=struct, iterations=1)
    #print(mask)
    mask_labels = measure.label(erodedMask,background=0,connectivity=1)
    pseudo_final = mask_labels == mask_labels[y,x,z]
    mask_final = ndimage.binary_dilation(pseudo_final, structure=struct2, iterations=1)
    show_canvas_sec(obtain_image_slice(mask_final))
    return (mask_final)

def algoritmo_pintor(imgA,imgB,alpha):

    return imgA * (1 - alpha) + imgB*alpha


def openWindowHeader(slices):
    idx = 0
    if SLICE_Z < len(slices):
        idx = SLICE_Z
    interfaz = [[sg.Text(slices[idx])]]
    col_interfaz = [[sg.Column(interfaz,scrollable=True,vertical_scroll_only=True)]] 
    ventanaheader = sg.Window("Header data",col_interfaz,size=(800,600),modal=True)
    while True:
        event, values = ventanaheader.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
    ventanaheader.close()
sg.theme("BluePurple")


layout1 = [[sg.Text("Seleccionar carpeta" ,size=(8,1)),sg.Input(key="-FOLDER-"),sg.FolderBrowse(),sg.Button("Ir"),
           sg.Text("Windowing"),sg.Input(key="wMin",size=(4,1)),sg.Input(key="wMax",size=(4,1)),sg.Button("Aplicar W"),sg.Button("Header")],]

layout2 = [[sg.Canvas(key='fig1'),sg.Canvas(key="fig2"),sg.Canvas(key="fig3")]]

sliders= [[sg.T('0',size=(4,1), key='-LEFT_x-'),
            sg.Slider((0,MAX_SLIDER_X), key='-SLIDER_X-', orientation='h', enable_events=True, disable_number_display=True),
            sg.T('0', size=(4,1), key='-RIGHT_x-'),
            sg.T('0',size=(4,1), key='-LEFT_y-'),
            sg.Slider((0,MAX_SLIDER_Y), key='-SLIDER_Y-', orientation='h', enable_events=True, disable_number_display=True),
            sg.T('0', size=(4,1), key='-RIGHT_y-'),
            sg.T('0',size=(4,1), key='-LEFT_z-'),
            sg.Slider((0,MAX_SLIDER_Z), key='-SLIDER_Z-', orientation='h', enable_events=True, disable_number_display=True),
            sg.T('0', size=(4,1), key='-RIGHT_Z-'),
            sg.Button("Cambiar slice"),
            sg.Button("Mostrar Hipotalamo"), sg.Button("Reset"), sg.Button("Segmentacion"),sg.Button("Mostrar atlas")]]

layout3 = [[sg.Canvas(key='fig4'),sg.Canvas(key="fig5"),sg.Canvas(key="fig6")]]

sliders2 = [[
            sg.T('0',size=(4,1), key='-LEFT_y2-'),
            sg.Slider((0,MAX_SLIDER_Y), key='-SLIDER_Y2-', orientation='h', enable_events=True, disable_number_display=True),
            sg.T('0', size=(4,1), key='-RIGHT_y2-'),
            sg.T('0',size=(4,1), key='-LEFT_z2-'),
            sg.Slider((0,MAX_SLIDER_Z), key='-SLIDER_Z2-', orientation='h', enable_events=True, disable_number_display=True),
            sg.T('0', size=(4,1), key='-RIGHT_Z2-'),
            sg.T('0',size=(4,1), key='-LEFT_x2-'),
            sg.Slider((0,MAX_SLIDER_X), key='-SLIDER_X2-', orientation='h', enable_events=True, disable_number_display=True),
            sg.T('0', size=(4,1), key='-RIGHT_x2-'),]]

layout = [
        [sg.Column(layout1, key='-COL1-',element_justification='c')],
        [sg.Column(sliders2,visible=True,key='sliders2'),
        [sg.Column(layout3, visible=True, key='-COL3-'),
        [sg.Column(layout2, visible=True, key='-COL2-'),
        [sg.Column(sliders,visible=True,key='sliders')]
        ]]]]


window = sg.Window("Practica 2",layout,size=(1920,1080),resizable=True)

phantom_data = np.zeros((2,2,2))
atlas_data = np.zeros((2,2,2))
dcm_data = np.zeros((512,512,61))
current_image = np.zeros((512,512))
pixel_len_mm = [5, 1, 1]
slices = None

def onclick(event):
    global SUB_POINT_1,SUB_POINT_2,MASK

    print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          ('double' if event.dblclick else 'single', event.button,
           event.x, event.y, event.xdata, event.ydata))
    if SUBIMAGE:
        if SUB_POINT_1 == None:
            SUB_POINT_1 = (int(event.xdata),int(event.ydata))
        elif SUB_POINT_2 == None:
            SUB_POINT_2 = (int(event.xdata),int(event.ydata))
            subimage()
    if SEGMENTATION:
        MASK = isocontorno(dcm_data,int(event.xdata),int(event.ydata),SLICE)
        show_canvas(obtain_image_slice(dcm_data))

while True:
    
    event,values = window.read()
    if event in (sg.WIN_CLOSED,"Exit"):
        break
    if event == "Ir":
        dcm_data,slices = load_dicom_folder(values["-FOLDER-"])
        phantom_data = load_dicom_file("data\icbm_avg_152_t1_tal_nlin_symmetric_VI.dcm")
        atlas_data = load_dicom_file("data\AAL3_1mm.dcm",atlas=True)

        MAX_SLIDER_X = dcm_data.shape[0]
        MAX_SLIDER_Y = dcm_data.shape[1]
        MAX_SLIDER_Z = dcm_data.shape[2]
        slider_x = window['-SLIDER_X-']
        slider_y = window['-SLIDER_Y-']
        slider_z = window['-SLIDER_Z-']
        slider_x.Update(range=(0, MAX_SLIDER_X))
        slider_y.Update(range=(0, MAX_SLIDER_Y))
        slider_z.Update(range=(0, MAX_SLIDER_Z))
        MAX_SLIDER_X2 = phantom_data.shape[0]
        MAX_SLIDER_Y2 = phantom_data.shape[1]
        MAX_SLIDER_Z2 = phantom_data.shape[2]
        slider_x2 = window['-SLIDER_X2-']
        slider_y2 = window['-SLIDER_Y2-']
        slider_z2 = window['-SLIDER_Z2-']
        slider_x2.Update(range=(0, MAX_SLIDER_X2))
        slider_y2.Update(range=(0, MAX_SLIDER_Y2))
        slider_z2.Update(range=(0, MAX_SLIDER_Z2))
        imgs = obtain_image_slice(dcm_data)
        show_canvas(imgs)
        imgs = obtain_image_slice(phantom_data)
        show_canvas(imgs,lower=True)


    if event == "Cambiar slice":
        SLICE_X = int(values["-SLIDER_X-"])
        SLICE_Y = int(values["-SLIDER_Y-"])
        SLICE_Z = int(values["-SLIDER_Z-"])
        SLICE_X2 = int(values["-SLIDER_X2-"])
        SLICE_Y2 = int(values["-SLIDER_Y2-"])
        SLICE_Z2 = int(values["-SLIDER_Z2-"])
        imgs = obtain_image_slice(dcm_data)
        show_canvas(imgs)
        imgs = obtain_image_slice(phantom_data,lower=True)
        show_canvas(imgs,lower=True)
        window.refresh()

    if event == "Aplicar W":
 
        data = apply_windowing(float(values['wMin']),float(values['wMax']))
        
        show_canvas_sec(data)
        window.refresh()

    if event == "Subimagen":
        SUBIMAGE = True
    if event == "Reset":
        SEGMENTATION = False
        imgs = obtain_image_slice(dcm_data)
        show_canvas(imgs)
        clean_canvas("fig_second")
        window.refresh()
    
    if event == "Segmentacion":
        SEGMENTATION = True
    if event == "Header":
        openWindowHeader(slices)
    if event == "Mostrar atlas":
        GENERAL_ATLAS = not GENERAL_ATLAS
        ONLY_HIPO = False
        imgs = obtain_image_slice(dcm_data)
        show_canvas(imgs)
        imgs = obtain_image_slice(phantom_data,lower=True)
        show_canvas(imgs,lower=True)
        window.refresh()
    if event == "Mostrar Hipotalamo":
        ONLY_HIPO = not ONLY_HIPO
        GENERAL_ATLAS = False
        imgs = obtain_image_slice(dcm_data)
        show_canvas(imgs)
        imgs = obtain_image_slice(phantom_data,lower=True)
        show_canvas(imgs,lower=True)
        window.refresh()

    window['-LEFT_x-'].update(int(values['-SLIDER_X-']))
    window['-RIGHT_x-'].update(int(MAX_SLIDER_X))
    window['-LEFT_y-'].update(int(values['-SLIDER_Y-']))
    window['-RIGHT_y-'].update(int(MAX_SLIDER_Y))
    window['-LEFT_z-'].update(int(values['-SLIDER_Z-']))
    window['-RIGHT_Z-'].update(int(MAX_SLIDER_Z))
    window['-LEFT_x2-'].update(int(values['-SLIDER_X2-']))
    window['-RIGHT_x2-'].update(int(MAX_SLIDER_X2))
    window['-LEFT_y2-'].update(int(values['-SLIDER_Y2-']))
    window['-RIGHT_y2-'].update(int(MAX_SLIDER_Y2))
    window['-LEFT_z2-'].update(int(values['-SLIDER_Z2-']))
    window['-RIGHT_Z2-'].update(int(MAX_SLIDER_Z2))
window.close()

