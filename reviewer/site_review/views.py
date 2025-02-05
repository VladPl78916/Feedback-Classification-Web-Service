from django.contrib.postgres.search import TrigramSimilarity
from django.urls import reverse, reverse_lazy
from django.views.generic import ListView, CreateView, DetailView
import plotly.graph_objects as go
from site_review.forms import ReviewForm
from site_review.models import Companies, ReviewStatistics
from .predictor_tool.predictor import load_model, predict_text
import torch
import dill
from googletrans import Translator


class HomePage(ListView):
    model = ReviewStatistics
    template_name = 'site_review/index.html'
    context_object_name = 'posts'


class Search(ListView):
    model = ReviewStatistics
    template_name = 'site_review/index.html'
    context_object_name = 'posts'

    def get_queryset(self):
        query = self.request.GET.get('do')
        print(type(query))

        similarity = (TrigramSimilarity('company_name', query))
        return (
            self.model.objects.annotate(similarity=similarity).filter(
                similarity__gt=0.1).order_by(
                '-similarity'))

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['title'] = 'Результаты поиска: '
        return context


class CreateReview(CreateView):
    model = Companies
    form_class = ReviewForm
    template_name = 'site_review/form.html'
    success_url = reverse_lazy('home')

    def form_valid(self, form):
        w = form.save(commit=False)
        w.company_name = w.company_name.lower().capitalize()
        model = load_model('site_review/predictor_tool/textcnn_model.pth')
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        with open('site_review/predictor_tool/vocab_dl.dill', 'rb') as f:
            vocab = dill.load(f)
        translator = Translator()
        translation = translator.translate(w.review, src='ru', dest='en')
        print(translation.text)
        prediction = predict_text(model, translation.text, vocab, device)
        print(prediction)
        w.type_review = prediction
        return super().form_valid(form)
    
    def get_initial(self):
        initial = super().get_initial()
        initial['company_name'] = self.request.GET.get('company', '')
        print(self.request.GET.get('company', ''))
        return initial


class Statistic(DetailView):
    model = ReviewStatistics
    template_name = 'site_review/statistic.html'
    context_object_name = 'post'
    slug_url_kwarg = 'post_slug'

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        vals = [self.object.positive_reviews, self.object.negative_reviews]
        labels = ['Положительные', 'Отриацтельные']
        colors = ['rgb(0, 112, 0)', 'rgb(187, 0, 0)']
        fig = go.Figure(data=go.Pie(
            labels=labels, values=vals, textinfo='none', marker=dict(colors=colors), ))
        fig.update_layout(width=672, height=573, legend=dict(
        orientation='h', 
        yanchor='bottom', 
        y=1.1,  
        xanchor='center', 
        x=0.5,
          font=dict(
            size=23, 
            color='black'
    )))
        graph_html = fig.to_html(full_html=False)
        context['diagram'] = graph_html
        return context

    def get_absolute_url(self):
        return reverse('post', kwargs={'post_slug': self.slug})