import matplotlib.pyplot as plt

# fig, ax = plt.subplots(figsize=(0.5, 3.5))
fig, ax = plt.subplots(figsize=(3.5, 0.5))

ax.text(0.5, 3, "Underexplored Region\n(higher return)",
        fontsize=25,
        ha='center', va='center',
        # rotation=90
        )

ax.set_axis_off()   # hide axes
plt.savefig('../paper_plots/labelText1.svg', format='svg', bbox_inches='tight')
plt.show()
