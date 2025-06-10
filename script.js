document.addEventListener('DOMContentLoaded', function() {
    // Handle hero section buttons
    const heroDonateFoodBtn = document.querySelector('.hero-content .btn-primary');
    const heroContributeMoneyBtn = document.querySelector('.hero-content .btn-secondary');
    
    if (heroDonateFoodBtn) {
        heroDonateFoodBtn.addEventListener('click', function() {
            window.location.href = 'donate-food.html';
        });
    }
    
    if (heroContributeMoneyBtn) {
        heroContributeMoneyBtn.addEventListener('click', function() {
            window.location.href = 'donate-money.html';
        });
    }
    
    // Handle category cards
    const categoryCards = document.querySelectorAll('.category-card');
    categoryCards.forEach(card => {
        card.addEventListener('click', function() {
            const button = this.querySelector('.category-btn');
            if (button) {
                const pagePath = button.getAttribute('data-page');
                window.location.href = pagePath;
            }
        });
    });
    
    // Handle category buttons (prevent propagation)
    const categoryButtons = document.querySelectorAll('.category-btn');
    categoryButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.stopPropagation(); // Prevent the card click from being triggered
            const pagePath = this.getAttribute('data-page');
            window.location.href = pagePath;
        });
    });
    
    // Handle payment option selection if on donate-money page
    const paymentOptions = document.querySelectorAll('.payment-option');
    paymentOptions.forEach(option => {
        option.addEventListener('click', function() {
            const parentGroup = this.parentElement;
            parentGroup.querySelectorAll('.payment-option').forEach(opt => {
                opt.classList.remove('selected');
            });
            this.classList.add('selected');
        });
    });
    
    // Handle amount option selection if on donate-money page
    const amountOptions = document.querySelectorAll('.amount-option');
    amountOptions.forEach(option => {
        option.addEventListener('click', function() {
            const parentGroup = this.parentElement;
            parentGroup.querySelectorAll('.amount-option').forEach(opt => {
                opt.classList.remove('selected');
            });
            this.classList.add('selected');
            
            // If custom option is selected, focus on the custom amount input
            if (this.textContent === 'Custom') {
                document.getElementById('custom-amount').focus();
            }
        });
    });
});