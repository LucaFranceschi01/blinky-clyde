# Team Information

**Course:** 24304 - Artificial Intelligence

**Semester:** Semester 1 - 2023

**Instructor:** Sergio Calo

**Team name:** blinky-clyde

**Team members:**

* 199149 - Luca Franceschi - luca.franceschi01@estudiant.upf.edu - LucaFranceschi01
* 198640 - Candela Álvarez López - candela.alvarez02@estudiant.upf.edu - CandelaAlvLop

## Why Blinky & Clyde?

Originally, Namco designed the ghosts to have their own distinct personalities.
Each ghost's name gives a hint to its strategy for tracking down Pac-Man: Shadow ("Blinky") always chases Pac-Man, Speedy ("Pinky") tries to get ahead of him, Bashful ("Inky") uses a more complicated strategy to zero in on him, and Pokey ("Clyde") alternates between chasing him and running away.
The ghosts' Japanese names are おいかけ, chase; まちぶせ, ambush; きまぐれ, fickle; and おとぼけ, playing dumb, respectively.

## What is our approach?

We have tried to approach this AI Pacman contest by trying out Approximate Q-Learning. After lots of struggling, we've implemented eight features with the following intentions:

- Invaders: We want our agents to try to eat the invaders when they are near and visible, and when we are not scared.
- Defenders: We want to avoid being near defenders when we are vulnerable in the opponent's field, but if at a certain point the enemies become scared, they better run!
- Food offense: We want our agents to eat all the food they can, without endangering themselves of course.
- Food defense: If we have few food left, we want to become more defensive, and protect the few food we still have.
- Capsules: If we are being chased, and in our way home we have some capsule near (under some certain conditions), we will try to eat it to make a comeback!
- Return: Eating food has no positive effect unless we return it to our field, so if we are full of food, we better return it before something bad happens.
- Explore: If we want to attack but there is a very good defender trying to intimidate us, the agents will try to split in separate parts of the map so that at least one of them gets free.
- Stop: We do not want lazy agents, so they will be penalized if they take the STOP action.

We have to admit that most of the time our approach works as intended, but sometimes it can be hard to debug why it is confused under some circumstances.

## Acknowledgements

We wanted to thank our instructor, this would've not been possible without all the help and assistance received from him.
