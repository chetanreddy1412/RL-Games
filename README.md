# RL-Games
This project as a part of the [RL Games Hackathon in Shaastra 2022](https://dare2compete.com/hackathon/rl-games-shaastra-2022-indian-institute-of-technology-iit-madras-244941). 

## Videos:
1. Video of the agent in action (Blue represents our agent while Red represents another participant in the tournament):
![Our Agent in a Tournament Game](https://github.com/chetanreddy1412/RL-Games/assets/60615610/4d3b1439-be8f-4aef-a992-0c36f62a2d71)

2. Video of the our trained agent playing against itself:
![Our Agent playing against a Copy of Itself](https://github.com/chetanreddy1412/RL-Games/assets/60615610/636d251b-37ed-441b-b260-1e867a93aa20)




## Implementation:
1. We created agents using Deep Reinforcement Learning(RL) to compete against each other in a virtual two-player 2D game setting.
2. We implemented two Deep RL methods, namely, deep Q networks and a vanilla policy gradient.
3. A novel feature engineering technique inspired by the decision tree approach was crafted to represent the game board and the state of the snakes.
4. We trained the agents in an iterative manner where a random agent plays against a semi-trained agent (from the previous iteration).
5. Our agents could dominate the score and win every game in the tournament. **With over 700 participants from across the country, we emerged as the Winners.**
![Winning Certificate](https://github.com/chetanreddy1412/RL-Games/blob/main/Winners%20Certificate.jpg)


To play against the trained bot:
1. Git clone the repository and install the required packages
2. Run the Human_vs_AI.py file
   ```bash
   python Human_vs_AI.py
   ```
3. One can either play against a random agent (to understand the enviroment and get used to the controls) or the trained agent.
4. About the environment:
- The Blue Snake is the player and the Red Snake is the trained agent
- Allowed Controls: Up Arrow Key, Left Arrow Key, Right Arrow Key
5. For more details about the environment, refer https://github.com/Xerefic/RL_Games


  


