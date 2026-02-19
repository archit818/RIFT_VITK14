with open('test_result.txt', 'r', encoding='utf-16-le') as f:
    lines = f.readlines()

# Print first 30 lines (summary + accounts)
for line in lines[:30]:
    print(line.rstrip())
