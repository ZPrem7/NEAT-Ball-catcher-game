import os
import pygame
import random
import neat
import time
import math
import pickle

pygame.font.init()

win_width = 700
win_height = 500

win = pygame.display.set_mode((win_width, win_height))
pygame.display.set_caption("Ball Catcher")

score_font = pygame.font.Font(None, 36)
ball_img = pygame.image.load(os.path.join("images", "ball.png"))
basket_img = pygame.image.load(os.path.join("images", "basket.png"))
background_image = pygame.image.load(os.path.join("images", "background.jpg")).convert_alpha()

score = 0
missed = 0
gen=0
game=1

class BASKET:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.img = basket_img
        self.width = self.img.get_width()
        self.height = self.img.get_height()
        self.speed = 30

    def play(self):
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            if self.x > 0:
                self.x -= self.speed
        if keys[pygame.K_RIGHT]:
            if self.x < win_width - self.img.get_width():
                self.x += self.speed

    def move_left(self):
        if self.x > 0 and (self.x-self.speed)>0:
            self.x -= self.speed

    def move_right(self):
        if self.x < win_width - self.img.get_width() and (self.x + self.speed)<(win_width - self.img.get_width()):
            self.x += self.speed

    def draw(self, win):
        win.blit(self.img, (self.x, self.y))


class BALL:
    def __init__(self, x):
        self.width = 20
        self.height = 20
        self.img = ball_img
        self.x = x
        self.y = 0
        self.speed = 15

    def move(self):
        self.y += self.speed

    def caught(self, player):
        return self.y + self.height >= player.y and self.x + self.width >= player.x and self.x <= player.x + player.width

    def missed(self):
        return self.y >= win_height

    def draw(self, win):
        win.blit(self.img, (self.x, self.y))






def play_game(g, config):
    global score,missed
    net = neat.nn.FeedForwardNetwork.create(g, config)
    player = BASKET((win_width - 50) // 2, win_height - 80)
    balls = []
    running = True
    clock = pygame.time.Clock()
    while running:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()
                quit()
        if len(balls) == 0:
            balls.append(BALL(random.randint(0, win_width - 30)))
        basket_pos_x = player.x + int(player.width / 2)
        basket_pos_y = win_height - player.img.get_height()
        ball_pos_x = balls[0].x + int(balls[0].img.get_width() / 2)
        ball_pos_y = balls[0].y + balls[0].img.get_height()


        """  
        ### To test the model comment the "player.play()" and uncomment the ai controls
        
        # ai_controls
        output = net.activate((basket_pos_x, ball_pos_x, abs(ball_pos_y - basket_pos_y)))
        if output[1] > output[2]:
            player.move_left()
        elif output[1] < output[2]:
            player.move_right()
        """
        player.play()




        win.blit(background_image, (0, 0))
        player.draw(win)
        for b in balls:
            b.move()
            b.draw(win)

        score_text = score_font.render("Score: {}".format(score), True, (255, 255, 255))
        win.blit(score_text, (10, 10))
        missed_text = score_font.render("Missed: {}".format(missed), True, (255, 255, 255))
        win.blit(missed_text, (win_width - missed_text.get_width() - 10, 10))
        pygame.display.update()

        if balls[0].caught(player):
            score += 1
            balls.remove(balls[0])
        if len(balls) != 0:
            if balls[0].missed():
                missed += 1
                balls = []




def draw_window(win, player, balls,gen):
    global score, missed
    win.blit(background_image, (0, 0))
    player.draw(win)
    for b in balls:
        b.move()
        b.draw(win)
    Gen_text = score_font.render("Gen: {}".format(gen), True, (255, 255, 255))
    win.blit(Gen_text, (300, 15))

    score_text = score_font.render("Score: {}".format(score), True, (255, 255, 255))
    win.blit(score_text, (10, 10))
    missed_text = score_font.render("Missed: {}".format(missed), True, (255, 255, 255))
    win.blit(missed_text, (win_width - missed_text.get_width() - 10, 10))
    pygame.display.update()



def train_ai(genomes, config):
    global score, missed ,gen
    game=1
    for g_id, g in genomes:
        g.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(g, config)
        player = BASKET((win_width - 50) // 2, win_height - 80)
        balls = []
        running = True
        clock = pygame.time.Clock()
        while running:
            #clock.tick(10)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    quit()
            if len(balls) == 0:
                balls.append(BALL(random.randint(0, win_width - 30)))
            basket_pos_x=player.x+int(player.width/2)
            basket_pos_y=win_height-player.img.get_height()
            ball_pos_x=balls[0].x+int(balls[0].img.get_width()/2)
            ball_pos_y=balls[0].y+balls[0].img.get_height()
            output = net.activate((basket_pos_x, ball_pos_x,abs(ball_pos_y-basket_pos_y)))
            if output[1] > output[2]:
                player.move_left()
            elif output[1] < output[2]:
                player.move_right()
            draw_window(win, player, balls,gen)

            if balls[0].caught(player):
                score += 1
                g.fitness += 5
                balls=[]
                game += 1
                break

            if len(balls) != 0:
                if balls[0].missed():
                    missed += 1
                    balls = []
                    game += 1
                    break


    gen+=1








def run_neat(config):
    #p=neat.Checkpointer.restore_checkpoint('neat-checkpoint-12')
    p = neat.Population(config)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(1))



    model_file = p.run(train_ai, 100)
    with open("model.pickle", "wb") as f:
       pickle.dump(model_file, f)



    """
    with open("model.pickle", "rb") as f:
        model_file=pickle.load(f)
    play_game(model_file,config)
    """




if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)
    run_neat(config)

