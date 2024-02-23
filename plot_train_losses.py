import matplotlib.pyplot as plt
input_file_path = 'train_losses.txt' # Replace with the path to your input file
output_file_path = 'listsoflosses.txt'  # Replace with the path to your output file

# Open the input file for reading and the output file for writing
with open(input_file_path, 'r') as input_file:
    # Iterate through each line in the input file
    dict ={}
    train_loss = []
    val_loss = []
    for line in input_file:
        # Check if the line contains any of the specified keywords
        # if "epoch" in line:
        #     # Write the line to the output file
        #     output_file.write(line)
        #     continue

        words = line.split()
        if "step" in line:
            train_loss.append(words[-1])
            continue
        if "loss" in line:
            val_loss.append(words[-1])

with open(output_file_path, 'w') as output_file:
    for i in train_loss:
        output_file.write(str(i) + " ")
    output_file.write("\n\n")
    for i in val_loss:
        output_file.write(str(i) + " ")



# Assuming you have lists representing epochs and corresponding train losses
steps = [1]
for i in range(52):
    if i == 0:
        continue
    steps.append(i*100)
print(steps)

plt.plot(steps, train_loss, 'b', label='Training Loss')
plt.plot(steps, val_loss, 'r', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.gca().invert_yaxis() 
plt.legend()
max_loss = max(max(train_loss), max(val_loss))

# Set the y-axis limit to the maximum loss value
plt.ylim(0, max_loss + 0.1)
plt.show()

        
            
