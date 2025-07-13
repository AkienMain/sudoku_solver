# Demo Video

[on desktop](https://www.youtube.com/watch?v=KSHXEJjfQM8)  
[on Google Colab](https://www.youtube.com/watch?v=Zx7rTNGdDCU)  

# Example
### Input
![input/sample.png](https://github.com/AkienMain/sudoku_solver/blob/master/input/sample.png?raw=true)  
### Output
![output/sample.png/output.png](https://github.com/AkienMain/sudoku_solver/blob/master/output/sample.png/output.png?raw=true)  

---

# Download
- ## Set project folder
  - example) C:\sudoku_solver-master
# Set configuration file
- /config.csv
  - corrected_image_size: Length of a side of corrected image
  - debug_level: How often print log
    - 0: None
    - 1: Low frequency
    - 2: High frequency

# Solve Sudoku (Desktop)
- ## Set image
  - Set input image in the "input" folder.
    - example) C:\sudoku_solver-master\input\input_image.png
- ## Run code
  - Open command prompt or PowerShell.
  - Set current directory to this project folder.
    - example)
        ```
        cd C:\sudoku_solver-master
        ```
  - Run this code.
    - example)
        ```
        python sudoku_solver input_image.png
        ```
    - If you run code without input image's name, this program import "sample.png" and solve it.
        ```
        python sudoku_solver
        ```
# Solve Sudoku (Google Colab)
Upload folder on your Google drive

![README_img/001.jpg](https://github.com/AkienMain/sudoku_solver/blob/master/README_img/001.jpg?raw=true)  

Execute script.ipynb

![README_img/002.jpg](https://github.com/AkienMain/sudoku_solver/blob/master/README_img/002.jpg?raw=true)  
![README_img/003.jpg](https://github.com/AkienMain/sudoku_solver/blob/master/README_img/003.jpg?raw=true)  
![README_img/004.jpg](https://github.com/AkienMain/sudoku_solver/blob/master/README_img/004.jpg?raw=true)  