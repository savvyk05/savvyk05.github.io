import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)

# Fonts
font = pygame.font.Font(None, 36)

# Classes
class Arrow(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((50, 10))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect(midbottom=(SCREEN_WIDTH // 2, SCREEN_HEIGHT - 30))

    def update(self):
        self.rect.centerx, _ = pygame.mouse.get_pos()

    def shoot(self):
        ball = Ball(self.rect.centerx, self.rect.top)
        all_sprites.add(ball)
        balls.add(ball)

class Ball(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.Surface((10, 10))
        self.image.fill(RED)  # Change color to red
        self.rect = self.image.get_rect(center=(x, y))

    def update(self):
        self.rect.y -= 5
        if self.rect.y < 0:
            self.kill()

class Shape(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        shape_type = random.choice(['circle', 'triangle', 'square'])
        self.image = pygame.Surface((30, 30), pygame.SRCALPHA)  # Set SRCALPHA for transparent background
        self.rect = self.image.get_rect(center=(random.randint(0, SCREEN_WIDTH), random.randint(-SCREEN_HEIGHT, 0)))

        if shape_type == 'circle':
            pygame.draw.circle(self.image, WHITE, (15, 15), 15)
        elif shape_type == 'triangle':
            pygame.draw.polygon(self.image, WHITE, [(15, 0), (0, 30), (30, 30)])
        elif shape_type == 'square':
            pygame.draw.rect(self.image, WHITE, (0, 0, 30, 30))
        self.rect = self.image.get_rect(center=(random.randint(0, SCREEN_WIDTH), random.randint(-SCREEN_HEIGHT, 0)))


    def update(self):
        self.rect.y += 2
        if self.rect.y > SCREEN_HEIGHT:
            self.kill()
            global lives
            lives -= 1

# Functions
def draw_text(text, font, color, surface, x, y):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.topleft = (x, y)
    surface.blit(text_surface, text_rect)

def start_menu():
    running = True
    while running:
        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    start_button_rect = pygame.Rect(300, 200, 200, 50)
                    exit_button_rect = pygame.Rect(300, 300, 200, 50)
                    if start_button_rect.collidepoint(mouse_pos):
                        return True
                    elif exit_button_rect.collidepoint(mouse_pos):
                        pygame.quit()
                        sys.exit()

        screen.fill(BLACK)

        # Draw "Shape Shooter" text
        draw_text("Shape Shooter", font, WHITE, screen, SCREEN_WIDTH // 2 - 100, 100)

        # Draw "Start" button
        start_button_rect = pygame.Rect(300, 200, 200, 50)
        if start_button_rect.collidepoint(mouse_pos):
            # Apply wobbling effect when mouse hovers over the button
            draw_text("Start", font, (255, 255, 255), screen, 400 + random.randint(-2, 2), 225 + random.randint(-2, 2))
        else:
            draw_text("Start", font, WHITE, screen, 400, 225)

        # Draw "Exit" button
        exit_button_rect = pygame.Rect(300, 300, 200, 50)
        if exit_button_rect.collidepoint(mouse_pos):
            # Apply wobbling effect when mouse hovers over the button
            draw_text("Exit", font, (255, 255, 255), screen, 400 + random.randint(-2, 2), 325 + random.randint(-2, 2))
        else:
            draw_text("Exit", font, WHITE, screen, 400, 325)

        pygame.display.update()

def game():
    max_shapes = 10  # Maximum number of shapes
    max_balls = 20  # Maximum number of balls
    global lives
    lives = 5
    score = 0
    game_over = False

    while not game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if pause_button_rect.collidepoint(pygame.mouse.get_pos()):  # Return to menu if pause button clicked
                        return
                    else:
                        arrow.shoot()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:  # Pause the game
                    return

        all_sprites.update()

        # Check for collisions between balls and shapes
        collisions = pygame.sprite.groupcollide(balls, shapes, True, False)
        for ball, shape in collisions.items():
            shape[0].kill()
            shapes.remove(shape[0])
            ball.kill()
            score += 1

        # Create new shapes
        if len(shapes) < max_shapes and random.randint(0, 100) < 2:
            new_shape = Shape()
            all_sprites.add(new_shape)
            shapes.add(new_shape)

        # Remove old balls
        while len(balls) > max_balls:
            balls.sprites()[0].kill()

        screen.fill(BLACK)
        all_sprites.draw(screen)

        # Draw pause button
        pygame.draw.rect(screen, WHITE, pause_button_rect)
        draw_text("Pause", font, BLACK, screen, 10, 10)

        # Draw score
        draw_text("Score: " + str(score), font, WHITE, screen, 10, 50)

        # Draw lives
        draw_text("Lives: " + str(lives), font, WHITE, screen, SCREEN_WIDTH - 150, 10)

        # Draw line at bottom
        pygame.draw.line(screen, WHITE, (0, SCREEN_HEIGHT - 30), (SCREEN_WIDTH, SCREEN_HEIGHT - 30), 2)

        pygame.display.flip()
        clock.tick(60)

        # Check if lives are zero
        if lives <= 0:
            game_over = True

# Initialize screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Shape Shooter")

# Sprites
all_sprites = pygame.sprite.Group()
arrows = pygame.sprite.Group()
balls = pygame.sprite.Group()
shapes = pygame.sprite.Group()

arrow = Arrow()
all_sprites.add(arrow)
arrows.add(arrow)

pause_button_rect = pygame.Rect(10, 10, 80, 30)

clock = pygame.time.Clock()

# Run the game
if start_menu():
    game()
