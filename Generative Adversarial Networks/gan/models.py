import torch


class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels, 128, 4, 2, 1),           # in_ch, out_ch, kernel_size, stride, padding
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128, 256, 4, 2, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(256, 512, 4, 2, 1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(512, 1024, 4, 2, 1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(1024, 1, 4, 1, 0),
        )

        ### EC2: Spectral Normalization
        self.ec2_layers = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(input_channels, 128, 4, 2, 1)),  # in_ch, out_ch, kernel_size, stride, padding
            torch.nn.LeakyReLU(0.2),
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(128, 256, 4, 2, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(256, 512, 4, 2, 1)),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.utils.spectral_norm(torch.nn.Conv2d(512, 1024, 4, 2, 1)),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(1024, 1, 4, 1, 0),
        )


        ##########       END      ##########

    def forward(self, x):

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        x = self.layers(x)
        # x = self.ec2_layers(x)

        ##########       END      ##########

        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        self.layers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(self.noise_dim, 1024, 4, 1, 0),  # in_ch, out_ch, kernel_size, stride, padding
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(1024, 512, 4, 2, 1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, output_channels, 4, 2, 1),
            torch.nn.Tanh(),
        )

        ### EC2: Spectral Normalization
        self.ec2_layers = torch.nn.Sequential(
            torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(self.noise_dim, 1024, 4, 1, 0)),  # in_ch, out_ch, kernel_size, stride, padding
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(),
            torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(1024, 512, 4, 2, 1)),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(512, 256, 4, 2, 1)),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.utils.spectral_norm(torch.nn.ConvTranspose2d(256, 128, 4, 2, 1)),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, output_channels, 4, 2, 1),
            torch.nn.Tanh(),
        )

        ##########       END      ##########

    def forward(self, x):

        ####################################
        #          YOUR CODE HERE          #
        ####################################

        x = self.layers(x)
        # x = self.ec2_layers(x)

        ##########       END      ##########

        return x
