from django.db import models
from django.urls import reverse
from slugify import slugify


class Companies(models.Model):
    company_name = models.CharField(
        max_length=100, verbose_name='Название компании')
    review = models.CharField(max_length=255, verbose_name='Отзыв')
    type_review = models.BooleanField()

    def save(self, *args, **kwargs):
        super().save(*args, **kwargs)
        stats = ReviewStatistics.objects.get_or_create(
            company_name=self.company_name)[0]
        # Обновление статистики
        stats.total_reviews += 1
        if self.type_review == 1:
            stats.positive_reviews += 1
        elif self.type_review == 0:
            stats.negative_reviews += 1
        stats.save()

    class Meta:
        verbose_name = 'Отзывы о компаниях'
        verbose_name_plural = 'Отзывы о компаниях'
        ordering = ['company_name']

    def __str__(self):
        return self.company_name


class ReviewStatistics(models.Model):
    company_name = models.CharField(
        max_length=100, verbose_name='Название компании', primary_key=True)
    total_reviews = models.IntegerField(default=0)
    negative_reviews = models.IntegerField(default=0)
    positive_reviews = models.IntegerField(default=0)
    slug = models.SlugField(
        max_length=255, db_index=True, unique=True, blank=True)

    class Meta:
        verbose_name = 'Статистика отзывов'
        verbose_name_plural = 'Статистика отзывов'

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = slugify(self.company_name,
                                lowercase=True, separator='-')
        super().save(*args, **kwargs)

    def __str__(self):
        return self.company_name

    def get_absolute_url(self):
        return reverse('statistic', kwargs={'post_slug': self.slug})
