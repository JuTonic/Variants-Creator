alpha = lambda n: n ** 3 + 12 * n ** 2 + 41 * n + 42

p1_0 = 42
p1_1 = 12
q1 = 42

for i in range(5):
    p1_0 *= alpha(i)
    p1_1 *= 12
    q1 *= alpha(i)

print(p1_0 - p1_1, q1)

p_o = 0

a = 1
b = 30
for i in range(1):
    a *= alpha(i)
    b *= 12
    p_n = (1 - p_o) * 12 / alpha(i)
    p_o = p_n + p_o

print(p_n)