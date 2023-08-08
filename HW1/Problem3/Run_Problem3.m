%% Load Notes and Music
clc; 
clear all; 
close all;
% You may reuse your 'load_data' function from problem 2
[smagNote,smagMusic,sphaseMusic] = load_data();
%% Compute The Transcribe Matrix: non-negative projection with gradient descent
% Use the 'transcribe_music_gradient_descent' function here
num_iter = 250;
threshold = 0;

lr1 = 0.0001;
[T1, E1, transMatT1, smagMusicProj1] = transcribe_music_gradient_descent(smagMusic, smagNote, lr1, num_iter, threshold);

lr2 = 0.001;
[T2, E2, transMatT2, smagMusicProj2] = transcribe_music_gradient_descent(smagMusic, smagNote, lr2, num_iter, threshold);

lr3 = 0.01;
[T3, E3, transMatT3, smagMusicProj3] = transcribe_music_gradient_descent(smagMusic, smagNote, lr3, num_iter, threshold);

lr4 = 0.1;
[T4, E4, transMatT4, smagMusicProj4] = transcribe_music_gradient_descent(smagMusic, smagNote, lr4, num_iter, threshold);

% Store final W for each eta value in a text file called "problem3b_eta_xxx.dat"
% where xxx is the actual eta value. E.g. for eta = 0.01, xxx will be "0.01".

save('./results/problem3b_eta_0.0001.mat','transMatT1');

save('./results/problem3b_eta_0.001.mat','transMatT2');

save('./results/problem3b_eta_0.01.mat','transMatT3');

save('./results/problem3b_eta_0.1.mat','transMatT4');

% Print the plot of E vs. iterations for each eta in a file called
% "problem3b_eta_xxx_errorplot.png", where xxx is the eta value.
% Print the eta vs. E as a bar plot stored in "problem3b_eta_vs_E.png".
figure(1);
plot(1:num_iter,E1,'b')
xlabel('The number of iteration');
ylabel('Error');
title('E vs. iterations');
saveas(gcf,'./results/problem3b_eta_0.0001.png')

figure(2);
plot(1:num_iter,E2,'b')
xlabel('The number of iteration');
ylabel('Error');
title('E vs. iterations');
saveas(gcf,'./results/problem3b_eta_0.001.png')

figure(3);
plot(1:num_iter,E3,'b')
xlabel('The number of iteration');
ylabel('Error');
title('E vs. iterations');
saveas(gcf,'./results/problem3b_eta_0.01.png')

figure(4);
plot(1:num_iter,E4,'b')
xlabel('The number of iteration');
ylabel('Error');
title('E vs. iterations');
saveas(gcf,'./results/problem3b_eta_0.1.png')

figure(5);
plot([lr1 lr2 lr3 lr4],[min(E1) min(E2) min(E3) min(E4)],'r')
xlabel('The learning rate eta');
ylabel('Error');
title('eta vs. E');
saveas(gcf,'./results/problem3b_eta_vs_E.png')

%% Synthesize Music
% You may reuse the 'synthesize_music' function from problem 2.
% write the synthesized music as 'polyushka_syn.wav' to the 'results' folder.
synthesizemusic = synthesize_music(sphaseMusic,smagMusicProj4)';
audiowrite('./results/polyushka_syn.wav',20000*synthesizemusic,16000);