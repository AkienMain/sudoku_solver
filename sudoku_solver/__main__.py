import functions as f
import sys

def main():
    args = sys.argv
    f.read_config(args)
    perspective = f.correct_distortion()
    f.recognize_sudoku_matrix()
    f.solve_sudoku()
    f.export_solution(perspective)

if __name__ == '__main__':
    main()
