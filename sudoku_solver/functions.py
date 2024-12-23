import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import cv2
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import variables as v
import keras
import itertools

def debug_print(string, end='\n', level=1):
    '''
    Print depending on debug level.

        Parameters:
            string : String
            end : Suffix string
            level : Dedug Level
    '''

    if v.debug_level >= level:
        print(string if string != None else '', end=end)

def get_filter_size(image, coef):
    '''
    Return filter size using image size as a reference.

        Parameters:
            image : Image
            coef : Coefficient Value

        Returns:
            filter_size : Filter size (odd integer)
    '''

    filter_size = int(min(image.shape[0], image.shape[1]) * coef / 2) * 2 + 1
    return filter_size

def get_contours(image):
    '''
    Return contours from image.

        Parameters:
            image : Image

        Returns:
            cnts : Contours
    '''

    filter_blur = get_filter_size(image, 0.01)
    filter_bin = get_filter_size(image, 0.1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (filter_blur, filter_blur))
    invert = cv2.bitwise_not(blur)
    binary = cv2.adaptiveThreshold(invert,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,filter_bin,1)
    cnts, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
    return cnts

def get_quadrangles(cnts):
    '''
    Return approximated contours.

        Parameters:
            cnts : Contours

        Returns:
            quadrangles : Quadrangles
    '''

    quadrangle_list = []
    for cnt in cnts:
        epsilon = 0.01*cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        # only quadrangle
        if len(approx) == 4:
            quadrangle_list.append(approx)
    quadrangles = tuple(quadrangle_list)
    return quadrangles

def align_quadrangle(quadrangle):
    '''
    Return aligned quadrangle.

        Parameters:
            quadrangle : Quadrangle before alignment

        Returns:
            quadrangle : Quadrangle after alignment
    '''

    # left to right
    quadrangle = quadrangle[np.argsort(quadrangle[:,0,0])]
    # lefttop : 0, teftbottom : 1
    if quadrangle[0,0,1] > quadrangle[1,0,1]:
        vertices_copy = quadrangle.copy()
        quadrangle[0] = vertices_copy[1].copy()
        quadrangle[1] = vertices_copy[0].copy()
    # righttop : 2, rightbottom : 3
    if quadrangle[2,0,1] > quadrangle[3,0,1]:
        vertices_copy = quadrangle.copy()
        quadrangle[2] = vertices_copy[3].copy()
        quadrangle[3] = vertices_copy[2].copy()
    return quadrangle

def transform_image(original, vertices):
    '''
    Return transformed image

        Parameters:
            original : Original image
            vertices : Vertices

        Returns:
            transformed : Transformed image
            perspective : Perspective
            perspective_reverse : Perspective for reverse
    '''

    cp = original.copy()
    input = np.float32(vertices.reshape(4, 2))
    output = np.float32([[0, 0], [0, v.corrected_image_size], [v.corrected_image_size, 0], [v.corrected_image_size, v.corrected_image_size]])
    perspective = cv2.getPerspectiveTransform(input, output)
    perspective_reverse = cv2.getPerspectiveTransform(output, input)
    transformed = cv2.warpPerspective(cp, perspective, (v.corrected_image_size, v.corrected_image_size)).astype(np.uint8)
    return transformed, perspective, perspective_reverse

def get_max_quadrangle(quadrangles, image, threshold):
    '''
    Return maximum contour

        Parameters:
            quadrangles : Quadrangles
            image : Image
            threshold : Ignore contours over this threshold (0<x<1)

        Returns:
            max_quad : Maximum quadrangle
    '''

    max_area = 0
    max_index = 0
    image_area = image.shape[0] * image.shape[1]
    for i in range(len(quadrangles)):
        area = cv2.contourArea(quadrangles[i])
        if area < image_area * threshold and \
            area > max_area:
            max_area = area
            max_index = i
    max_quad = quadrangles[max_index]
    return max_quad

def correct_distortion():
    '''
    Correct distortion

        Returns:
            perspective_tuple : Perspective
    '''

    image = cv2.imread(v.input_image_path)
    cnts = get_contours(image)
    quadrangles = get_quadrangles(cnts)
    quadrangle = get_max_quadrangle(quadrangles, image, 0.9)
    quadrangle = align_quadrangle(quadrangle)
    transformed, perspective, perspective_reverse = transform_image(image, quadrangle)
    cv2.imwrite(v.corrected_path, transformed)

    return (perspective, perspective_reverse)

def get_number_matrix(image):
    '''
    Get number matrix

        Parameters:
            image : Image

        Returns:
            num_mat : Number matrix
    '''

    model = keras.models.load_model(v.model_path)
    if v.debug_level >= 2:
        model.summary()
    input_size = model.layers[0].input.shape[1]

    fig, axs = plt.subplots(9, 9)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pad = int(v.corrected_image_size/(9*8))
    pad2 = 10
    im_mat = np.zeros((9*9, input_size, input_size), dtype=float)
    l = int(v.corrected_image_size / 9)
    num_mat = np.zeros((9, 9), dtype=int)
    for i in range(9):
        for j in range(9):
            im = gray[l*i+pad:l*(i+1)-pad, l*j+pad:l*(j+1)-pad]
            im = cv2.resize(im,(input_size-pad2*2,input_size-pad2*2))
            im = cv2.medianBlur(im, 3)
            im = cv2.adaptiveThreshold(im,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,255,40)
            im = cv2.bitwise_not(im)
            im = cv2.copyMakeBorder(im,pad2,pad2,pad2,pad2,cv2.BORDER_CONSTANT,(0))
            im = cv2.erode(im, (3, 3))
            im = im / 255.0
            im = im.astype(float)
            im_mat[i*9+j,:,:] = im

            if v.debug_level >= 1:
                axs[i, j].imshow(im_mat[i*9+j,:,:], cmap='gray')
                axs[i, j].axis('off')

    onehot = model.predict(im_mat, verbose=0)
    for i in range(9):
        for j in range(9):
            num_mat[i,j] = np.argmax(onehot[i*9+j])

    if v.debug_level >= 1:
        fig.savefig(v.separated_path)

    return num_mat

def recognize_sudoku_matrix():
    '''
    Correct distortion
    '''

    src = cv2.imread(v.corrected_path)
    num_mat = get_number_matrix(src)
    np.savetxt(v.matrix_read_path, num_mat, fmt='%i', delimiter=",")

def method_c_n_numbers(candidate_group, n):
    '''
    When there are only n possible numbers in n spaces, remove the candidate numbers from the other spaces.

        Parameters:
            candidate_group : Candidate group (row, columns or block)
            n : Numbers of numbers

        Returns:
            satisfies : Whether this situation satisfies the condition that there are only n possible numbers in n spaces
            pos_array : Posision Array
            num_array : Number Array
    '''

    satisfies = False
    pos_array = np.array([])
    num_array = np.array([])

    for pos in itertools.combinations(range(9), n):

        single_condition = True
        for pos_i in pos:
            if not candidate_group[pos_i].sum() in (range(2,n+1)):
                single_condition = False
        
        disjunction = np.full(9, False)
        for pos_i in pos:
            disjunction += candidate_group[pos_i]

        if disjunction.sum() == n and single_condition:
            satisfies = True
            pos_array = np.asarray(pos)
            num_array = np.argwhere(disjunction==True).reshape(n)
    
    return satisfies, pos_array, num_array

def calc_candidate(answer):
    '''
    Calculate candidates.

        Parameters:
            answer : Answer Matrix

        Returns:
            candidate : Candidate Matrix
    '''

    candidate = np.full((9,9,9), True)

    # (A) Eliminate the candidates of that number
    for i in range(9):
        for j in range(9):
            num = answer[i,j]
            if num != 0:
                candidate[i,j,:] = False # Already identified
                candidate[i,:,num-1] = False # Horizontal
                candidate[:,j,num-1] = False # Vertical
                candidate[i//3*3:i//3*3+3,j//3*3:j//3*3+3,num-1] = False # Block
                candidate[i,j,num-1] = True
                debug_print(f'[{v.c_fun}Method-A{v.c_f}] Position:{v.c_pos}{i+1}{j+1}{v.c_f}, Number:{v.c_num}{num}{v.c_f}', level=2)
    
    # (B) Focus on each number
    for gr in range(9):
        for num in range(1,10):

            # Horizontal
            candidate_group = candidate[gr,:,num-1].reshape(9)
            if candidate_group.sum() == 1:
                pos = np.where(candidate_group==True)[0][0]
                candidate[gr,pos,:] = False
                candidate[gr,pos,num-1] = True
                debug_print(f'[{v.c_fun}Method-B{v.c_f}] Position:{v.c_pos}{gr+1}{pos+1}{v.c_f}, Number:{v.c_num}{num}{v.c_f}, Row   [{v.c_gr}{gr+1}{v.c_f}]', level=2)
            
            # Vertical
            candidate_group = candidate[:,gr,num-1].reshape(9)
            if candidate_group.sum() == 1:
                pos = np.where(candidate_group==True)[0][0]
                candidate[pos,gr,:] = False
                candidate[pos,gr,num-1] = True
                debug_print(f'[{v.c_fun}Method-B{v.c_f}] Position:{v.c_pos}{pos+1}{gr+1}{v.c_f}, Number:{v.c_num}{num}{v.c_f}, Column[{v.c_gr}{gr+1}{v.c_f}]', level=2)
            
            # Block
            candidate_group = candidate[gr//3*3:gr//3*3+3,gr%3*3:gr%3*3+3, num-1].reshape(9)
            if candidate_group.sum() == 1:
                pos = np.where(candidate_group==True)[0][0]
                candidate[gr//3*3+pos//3, gr%3*3+pos%3,:] = False
                candidate[gr//3*3+pos//3, gr%3*3+pos%3,num-1] = True
                debug_print(f'[{v.c_fun}Method-B{v.c_f}] Position:{v.c_pos}{gr//3*3+pos//3+1}{gr%3*3+pos%3+1}{v.c_f}, Number:{v.c_num}{num}{v.c_f}, Block[{v.c_gr}{gr//3+1}{gr%3+1}{v.c_f}]', level=2)
    
    # (C) Focus on n numbers
    for n in range(2, 9):

        for gr in range(9):

            # Horizontal
            satisfies, pos_array, num_array = method_c_n_numbers(candidate[gr,:,:], n)
            if satisfies:
                for pos in range(9):
                    if not pos in pos_array:
                        candidate[gr, pos, num_array] = False
                pos_array_print = ((gr+1)*10+(pos_array+1))
                debug_print(f'[{v.c_fun}Method-C{v.c_f}] Position:{v.c_pos}{pos_array_print}{v.c_f}, Number:{v.c_num}{num_array+1}{v.c_f}, Row   [{v.c_gr}{gr+1}{v.c_f}], {n}-Numbers', level=2)
            
            # Vertical
            satisfies, pos_array, num_array = method_c_n_numbers(candidate[:,gr,:], n)
            if satisfies:
                for pos in range(9):
                    if not pos in pos_array:
                        candidate[pos, gr, num_array] = False
                pos_array_print = ((pos_array+1)*10+(gr+1))
                debug_print(f'[{v.c_fun}Method-C{v.c_f}] Position:{v.c_pos}{pos_array_print}{v.c_f}, Number:{v.c_num}{num_array+1}{v.c_f}, Column[{v.c_gr}{gr+1}{v.c_f}], {n}-Numbers', level=2)

            # block
            satisfies, pos_array, num_array = method_c_n_numbers(candidate[gr//3*3:gr//3*3+3,gr%3*3:gr%3*3+3,:].reshape(9,9), n)
            if satisfies:
                for pos in range(9):
                    if not pos in pos_array:
                        candidate[gr//3*3+pos//3, gr%3*3+pos%3, num_array] = False
                pos_array_print = ((gr//3*3+pos_array//3+1)*10+(gr%3*3+pos_array%3+1))
                debug_print(f'[{v.c_fun}Method-C{v.c_f}] Position:{v.c_pos}{pos_array_print}{v.c_f}, Number:{v.c_num}{num_array+1}{v.c_f}, Block[{v.c_gr}{gr//3+1}{gr%3+1}{v.c_f}], {n}-Numbers', level=2)
    
    return candidate

def identify(candidate):
    '''
    Identify answer matrix from candidate matrix.

        Parameters:
            candidate : Candidate Matrix

        Returns:
            answer : Answer Matrix
    '''

    answer = np.zeros((9,9), dtype=int)
    for i in range(9):
        for j in range(9):
            if candidate[i,j,:].sum() == 1:
                num = np.where(candidate[i,j,:]==True)[0][0] + 1
                answer[i,j] = num
                candidate[i,j,num-1] = False
                debug_print(f'[{v.c_fun}Identify{v.c_f}] Position:{v.c_pos}{i+1}{j+1}{v.c_f}, Number:{v.c_num}{num}{v.c_f}', level=2)
    return answer

def check_completeness(answer, candidate):
    '''
    Check completeness

        Parameters:
            answer : Answer Matrix
            candidate : Candidate Matrix

        Returns:
            is_completed : Whether answer is completed
            is_consistent : Whether answer is consistent
    '''

    debug_print(f'{v.c_fun}--- Check Completeness ---{v.c_f}')

    is_completed = True
    is_consistent = True

    for gr in range(9):
        if 0 in answer[gr,:] or answer[gr,:].sum() != 45:
            is_completed = False
            debug_print(f'[{v.c_fun}Check Completeness{v.c_f}] Row[{v.c_gr}{gr+1}{v.c_f}]: {v.c_num}{answer[gr,:]}{v.c_f}', level=2)
    
    for gr in range(9):
        if 0 in answer[:,gr] or answer[:,gr].sum() != 45:
            is_completed = False
            debug_print(f'[{v.c_fun}Check Completeness{v.c_f}] Column[{v.c_gr}{gr+1}{v.c_f}]: {v.c_num}{answer[:,gr]}{v.c_f}', level=2)
    
    for gr in range(9):
        if 0 in answer[gr//3*3:gr//3*3+3,gr%3*3:gr%3*3+3] or answer[gr//3*3:gr//3*3+3,gr%3*3:gr%3*3+3].sum() != 45:
            is_completed = False
            debug_print(f'[{v.c_fun}Check Completeness{v.c_f}] Block[{v.c_gr}{gr//3+1}{gr%3+1}{v.c_f}]: {v.c_num}{answer[gr//3*3:gr//3*3+3,gr%3*3:gr%3*3+3].reshape(9)}{v.c_f}', level=2)

    for i in range(9):
        for j in range(9):
            if answer[i,j] == 0 and candidate[i,j,:].sum() == 0:
                is_consistent = False
    
    debug_print(f'[{v.c_fun}Check Completeness{v.c_f}]', end='')
    if is_completed:
        debug_print(' Completed!', end='')
    else:
        debug_print(' NOT Completed!', end='')
    if not is_consistent:
        debug_print(', Inconsistent!', end='')
    debug_print('')
    
    return is_completed, is_consistent

def contrarian_law(answer, candidate):
    '''
    run contrarian law method

        Parameters:
            answer : Answer Matrix
            candidate : Candidate Matrix

        Returns:
            answer : Answer Matrix
            candidate : Candidate Matrix
    '''

    debug_print(f'{v.c_fun}--- Contrarian Law Solution ---{v.c_f}')
    
    breaks_loop = False

    for i in range(9):
        for j in range(9):
            
            if candidate[i,j,:].sum() == 2:
                hypo_nums = np.argwhere(candidate[i,j,:]==True).reshape(2)+1
                for hypo_num in hypo_nums:

                    debug_print(f'[{v.c_fun}Contrarian Law Solution{v.c_f}] Hypothesis, Position:{i+1}{j+1}, Number:{hypo_num}')
                    answer_tmp = np.copy(answer)
                    candidate_tmp = np.copy(candidate)
                    answer_tmp[i,j] = hypo_num
                    answer_tmp, candidate_tmp = rulebased(answer_tmp, candidate_tmp)
                    is_completed, is_consistent = check_completeness(answer_tmp, candidate_tmp)

                    if is_completed and is_consistent:
                        answer = answer_tmp
                        candidate = candidate_tmp
                        breaks_loop = True
                        break
                    
            if breaks_loop:
                break
        if breaks_loop:
            break
    
    return answer, candidate


def rulebased(answer, candidate):
    '''
    run rule-based method

        Parameters:
            answer : Answer Matrix
            candidate : Candidate Matrix

        Returns:
            answer : Answer Matrix
            candidate : Candidate Matrix
    '''

    is_updated_candidate = True
    is_updated_answer = True
    debug_print(f'{v.c_fun}--- Rule-based Solution ---{v.c_f}')
    
    while is_updated_candidate or is_updated_answer:
        candidate_new = calc_candidate(answer)
        answer_new = identify(candidate_new)
        is_updated_candidate = not np.array_equal(candidate, candidate_new)
        is_updated_answer = not np.array_equal(answer, answer_new)

        debug_print(f'[{v.c_fun}Rule-based Solution{v.c_f}]', end='')
        if is_updated_answer:
            debug_print(' Answer is Updated!', end='')
        if is_updated_candidate:
            debug_print(' Candidates are Updated!', end='')
        if not is_updated_answer and not is_updated_candidate:
            debug_print(' Finished!', end='')
        debug_print('')

        candidate = candidate_new
        answer = answer_new
    
    return answer, candidate

def solve_sudoku():
    '''
    Solve Sudoku
    
    '''

    # Answer Matrix
    answer = np.loadtxt(v.matrix_read_path, dtype=int, delimiter=',')

    # Candidate Matrix
    candidate = np.full((9,9,9), True)

    answer, candidate = rulebased(answer, candidate)
    is_completed, _ = check_completeness(answer, candidate)

    if not is_completed:
        answer, candidate = contrarian_law(answer, candidate)
        is_completed, _ = check_completeness(answer, candidate)
    
    np.savetxt(v.matrix_solved_path, answer, fmt='%i', delimiter=",")

def overwrite_answer(base_image, matrix_read, matrix_solved):
    '''
    Overwrite answer

        Parameters:
            base_image : Matrix (9*9) of base image
            matrix_read : Matrix (9*9) read
            matrix_solved : Matrix (9*9) solved

        Returns:
            image : List of matrix
    '''
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    l = int(v.corrected_image_size/9)
    offset = int(l/4)
    fontScale = 6
    thickness = 12

    for i in range(9):
        for j in range(9):
            position = (l*j+offset, l*(i+1)-offset)
            color = (0, 0, 255)
            if matrix_read[i,j] != 0:
                color = (255, 0, 0)
            image = cv2.putText(base_image, str(matrix_solved[i,j]), position, font, fontScale, color, thickness, cv2.LINE_AA)

    return image

def export_solution(perspective):
    '''
    Export solution

        Parameters:
            perspective : Perspective
    '''

    original = cv2.imread(v.input_image_path)
    matrix_read = np.loadtxt(v.matrix_read_path, dtype=int, delimiter=',')
    matrix_solved = np.loadtxt(v.matrix_solved_path, dtype=int, delimiter=',')

    size = (original.shape[1], original.shape[0])
    corrected = cv2.warpPerspective(original, perspective[0], (v.corrected_image_size, v.corrected_image_size)).astype(np.uint8)
    corrected_zeros = np.zeros(corrected.shape)

    numbers_cor_transp = overwrite_answer(corrected_zeros, matrix_read, matrix_solved)
    numbers_transp = cv2.warpPerspective(numbers_cor_transp, perspective[1], size).astype(np.uint8)
    numbers_transp_gray = cv2.cvtColor(numbers_transp,cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(numbers_transp_gray, 1, 255, cv2.THRESH_BINARY)

    numbers_cor = overwrite_answer(corrected, matrix_read, matrix_solved)
    numbers = cv2.warpPerspective(numbers_cor, perspective[1], size).astype(np.uint8)

    back = cv2.bitwise_and(original, original, mask=cv2.bitwise_not(mask))
    fore = cv2.bitwise_and(numbers, numbers, mask=mask)

    solution = back + fore
    cv2.imwrite(v.output_path, solution)

def read_config(args):
    '''
    Read config file

        Parameters:
            args : Command line arguments
    '''

    if len(args) >= 2:
        v.input_image_name = args[1]
    else:
        v.input_image_name = 'sample.png'


    v.project_path = os.getcwd()

    config = pd.read_csv(v.config_path, header=None, index_col=0)
    v.corrected_image_size = int(config.loc['corrected_image_size'].iloc[0])
    v.debug_level = int(config.loc['debug_level'].iloc[0])
    debug_print(f'{v.c_c}Debug Level: {v.debug_level}{v.c_f}')

    v.input_image_path = f'{v.project_path}/input/{v.input_image_name}'
    v.output_folder = f'{v.project_path}/output/{v.input_image_name}'

    if not os.path.isdir(v.output_folder):
        os.makedirs(v.output_folder)

    v.corrected_path = f'{v.output_folder}/corrected_distortion.png'
    v.separated_path = f'{v.output_folder}/separated_images.png'
    v.matrix_read_path = f'{v.output_folder}/matrix_read.csv'
    v.matrix_solved_path = f'{v.output_folder}/matrix_solved.csv'
    v.output_path = f'{v.output_folder}/output.png'