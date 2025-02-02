from django import forms
from .models import Companies


class ReviewForm(forms.ModelForm):
    class Meta:
        model = Companies
        fields = ['company_name', 'review']
        widgets = {'review': forms.Textarea(attrs={'cols': 50, 'rows': 5})}



