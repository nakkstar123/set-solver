import itertools

def points_form_line(p1, p2, p3):
    # Ensure the points are distinct (if not, we don’t have three different points on a line)
    if p1 == p2 or p1 == p3 or p2 == p3:
        return False

    # Check that the coordinate-wise sum (modulo 3) is (0,0,0,0)
    for i in range(4):
        if (p1[i] + p2[i] + p3[i]) % 3 != 0:
            return False
    return True

def find_line_indices(points):
    line_indices = []
    
    # Iterate over all combinations of 3 distinct indices.
    for i, j, k in itertools.combinations(range(len(points)), 3):
        if points_form_line(points[i], points[j], points[k]):
            line_indices.append((i+1, j+1, k+1))
    
    return line_indices