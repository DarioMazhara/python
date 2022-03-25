import math

def minimax(curr_depth, node_index, max_turn,
            scores, target_depth):
    if (curr_depth == target_depth):
        return scores[node_index]
    
    if (max_turn):
        return max(minimax(curr_depth + 1, node_index * 2,
                    False, scores, target_depth),
                    minimax(curr_depth + 1, node_index * 2 + 1,
                    False, scores, target_depth))
    else:
        return min(minimax(curr_depth + 1, node_index * 2,
                    True, scores, target_depth),
                   minimax(curr_depth + 1, node_index * 2 + 1,
                    True, scores, target_depth))
        

scores = [3, 5, 6, 3, 8, 2, 2]

tree_depth = math.log(len(scores), 2)
print (minimax(0, 0, True, scores, tree_depth))