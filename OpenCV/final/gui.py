import tkinter as tk
from PIL import Image, ImageTk
 

root = tk.Tk()
root.title('Refocus')
root.geometry('800x800')

img = Image.open('./images/output1.jpg')
img = img.resize((750, 500))
img = ImageTk.PhotoImage(img)
label = tk.Label(root, image=img, width=750, height=500)
label.place(x = 25, y = 25)

bar = tk.Scale(root, orient=tk.HORIZONTAL)
# bar.place(x = )

root.mainloop()
