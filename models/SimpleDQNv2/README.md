# SimpeDQNv2

It is first trained model for DQNChessEngine.

## Concept

Input consisted of single 384 element one dimensional array 
representing a board state that were valued by a network. 
Each square was represented by 6 states related to chess pieces and encoded by 1-of-N. 
It resulted in following codes:

| Piece | Code |
| :---: |:----:|
| None | [ 0, 0, 0, 0, 0, 0 ] |
| Pawn | [ 1, 0, 0, 0, 0, 0 ] |
| Knight | [ 0, 1, 0, 0, 0, 0 ] |
| Bishop | [ 0, 0, 1, 0, 0, 0 ] |
| Rook | [ 0, 0, 0, 1, 0, 0 ] |
| Queen | [ 0, 0, 0, 0, 1, 0 ] |
| King | [ 0, 0, 0, 0, 0, 1 ] |

If a piece standing on the encoded field was owned by the opponent,
then code was multiplied by -1.

## Structure

It is sequential model with three dense layers:
* First - 256 nodes
* Second - 512 nodes
* Third - 1 node

All of them were activated using ReLU function.

## Training

Adam was used as an optimiser and mean square error function represented loss.
During training two networks were used - active and target. Target network's weights were updated
by using soft updating (target_weights = theta * active_weights + (1 - theta) * target_weights).
There was constant probability epsilon that random move would be made in order to explore more possible states.
Training started after accumulating 96 (3 * batch size) states and their prizes.


### Metaparameters

* Training steps - 200k
* Memory buffer records - 100k
* Training batch size - 32
* Gamma - 0.99
* Theta - 0.005
* Constant epsilon - 0.3