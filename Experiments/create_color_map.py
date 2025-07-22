start = 1.0
end = 15.0
step = 0.1
n_steps = int((end - start) / step) + 1

for i in range(n_steps):
    value = round(start + i * step, 1)
    blue = int(255 * (i / (n_steps - 1)))  # Linear ramp to 255
    print(f"{value:.1f} val{value:.1f} 0 0 {blue} 0")
