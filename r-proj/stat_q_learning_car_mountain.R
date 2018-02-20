require(ggplot2)

episode_stat <- read.csv('../data/q_learning_car_mountain.csv')



# 奖励趋势
p <- ggplot(episode_stat, aes(x=episode, y=reward, color = feature))
p <- p + geom_line()
p

# 每轮步数趋势
p <- ggplot(episode_stat, aes(x=episode, y=length, color = feature))
p <- p + geom_line()
p



# 平滑曲线，方便观察
feature_method <- c('Pass', 'RBF', 'Scale')


fit_result <- data.frame()
for(feature in feature_method) {
  index <- episode_stat$feature == feature
  x <- episode_stat$episode[index]
  y <- episode_stat$reward[index]
  fit <- loess(y~x)
  fit_y <- predict(fit)
  
  this_result <- data.frame(feature=paste(feature,'fitted'), episode = x, reward = fit_y)
  fit_result <- rbind(fit_result, this_result)
}

p <- ggplot()
p <- p + geom_line(data=episode_stat, aes(x=episode, y=reward,  color = feature))
p <- p + geom_line(data=fit_result, aes(x=episode,y=reward, color=feature), size =2)
p <- p + xlab('Episode') + ylab('奖励') + ggtitle('学习曲线')
p

 
