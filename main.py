import pygame
import os
import random
import neat
import sys
import pickle

pygame.init()
screen_width = 1200
screen_height = 1200
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Box Evasion")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 28)


# Constants
DIR_HOLD_TIME = 20
BOX_SPEED = 8
ANT_SPEED = 8
SHRINK_FACTOR = .9995

class Ant:
    def __init__(self, x_pos, y_pos, size=10):
        self.X_POS = x_pos
        self.Y_POS = y_pos
        self.SIZE = size
        self.color = (125, 121, 120)
        self.ant_rect = pygame.Rect(self.X_POS, self.Y_POS, self.SIZE, self.SIZE)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.ant_rect)

    def move(self, dx, dy):
        self.ant_rect.x += dx
        self.ant_rect.y += dy

class Container:
    def __init__(self, x_pos, y_pos, size):
        self.X_POS = x_pos
        self.Y_POS = y_pos
        self.SIZE = size
        self.color = (171, 166, 164)
        self.box_rect = pygame.Rect(self.X_POS, self.Y_POS, self.SIZE, self.SIZE)

    def draw(self, screen, thickness=2):
        pygame.draw.rect(screen, self.color, self.box_rect, thickness)





def evaluate_genomes(genomes, config):    
    ants, nets, ge = [], [], []
    box = Container(
        random.randint(100, 600),
        random.randint(100, 600),
        size=500
    )

    dir_timer = 0
    vx, vy = 0, 0
    
    # place every ant inside that box
    for gid, genome in genomes:
        genome.fitness = 0
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        start_box_x = random.randint(box.X_POS + 10, box.X_POS + box.SIZE - 60)
        start_box_y = random.randint(box.Y_POS + 10, box.Y_POS + box.SIZE - 60)
        ants.append(Ant(start_box_x, start_box_y))
        ge.append(genome)

    prev_box_pos = (box.X_POS, box.Y_POS)
    run = True
    while run and ants:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()

        # shrink & move box but gradually move and move with time intervals
        box.SIZE = max(int(box.SIZE * SHRINK_FACTOR), 60)
        box.box_rect.size = (box.SIZE, box.SIZE)

        if dir_timer == 0:
            vx = random.choice((-1, 0, 1)) * BOX_SPEED
            vy = random.choice((-1, 0, 1)) * BOX_SPEED
            if vx == 0 and vy == 0:
                vx = BOX_SPEED
            dir_timer = DIR_HOLD_TIME
        dir_timer -= 1

        box.box_rect.x = max(0, min(screen_width  - box.SIZE, box.box_rect.x + vx))
        box.box_rect.y = max(0, min(screen_height - box.SIZE, box.box_rect.y + vy))

        cur_box_pos = (box.box_rect.x, box.box_rect.y)
        box_dx = (cur_box_pos[0] - prev_box_pos[0]) / BOX_SPEED   #normalize
        box_dy = (cur_box_pos[1] - prev_box_pos[1]) / BOX_SPEED   #normalize 
        prev_box_pos = cur_box_pos


        for i, ant in enumerate(ants):
            #get distances to box edges
            left = ant.ant_rect.x - box.box_rect.x
            right = (box.box_rect.x + box.SIZE) - (ant.ant_rect.x + ant.SIZE)
            top = ant.ant_rect.y - box.box_rect.y
            bottom = (box.box_rect.y + box.SIZE) - (ant.ant_rect.y + ant.SIZE)

            inputs = [
                left / box.SIZE * 2 - 1,
                right / box.SIZE * 2 - 1,
                top / box.SIZE * 2 - 1,
                bottom / box.SIZE * 2 - 1,
                box.SIZE / 500,
                box_dx / 5,
                box_dy / 5,]

            dx, dy = nets[i].activate(inputs)
            ant.move(int(dx * ANT_SPEED), int(dy * ANT_SPEED))

            #hit box edge remove them
            if min(left, right, top, bottom) < 0:
                ge[i].fitness -= 5
                ants.pop(i)
                nets.pop(i)
                ge.pop(i)
                continue

            ge[i].fitness +=  .5+0.1 * min(left, right, top, bottom) / box.SIZE



        screen.fill((48, 46, 46))
        for ant in ants:
            ant.draw(screen)
        box.draw(screen)

        # HUD
        best_now = max(g.fitness for g in ge) if ge else 0
        gen_lbl   = font.render(f"Gen: {p.generation}", True, (0, 0, 0))
        fit_lbl   = font.render(f"Best fitness: {best_now:.1f}", True, (0, 0, 0))
        screen.blit(gen_lbl, (10, 10))
        screen.blit(fit_lbl, (10, 40))



        pygame.display.update()
        clock.tick(30)


def run_neat(config):
    global p
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config
    )
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    p.add_reporter(neat.StatisticsReporter())
    winner = p.run(evaluate_genomes, 10)
    print(f"\nBest genome:\n{winner}")

    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run_neat(config_path)
    pygame.quit()