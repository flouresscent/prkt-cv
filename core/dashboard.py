import tkinter as tk
from tkinter import ttk

class StatusDashboard:
    def __init__(self, window_name="Parking Status"):
        self.root = tk.Tk()
        self.root.title(window_name)
        self.labels = {}

    def update(self, status_dict):
        for slot_id, is_free in status_dict.items():
            text = f"{slot_id}: {'Free' if is_free else 'Occupied'}"
            color = "green" if is_free else "red"

            if slot_id not in self.labels:
                lbl = ttk.Label(self.root, text=text, foreground=color, font=("Courier", 11))
                lbl.pack(anchor="w", padx=10, pady=1)
                self.labels[slot_id] = lbl
            else:
                self.labels[slot_id].config(text=text, foreground=color)

        self.root.update()

    def close(self):
        self.root.destroy()