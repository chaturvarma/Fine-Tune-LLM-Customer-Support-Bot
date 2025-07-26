categories = [
    "ORDER", "SHIPPING", "CANCEL", "INVOICE",
    "PAYMENT", "REFUND", "FEEDBACK", "CONTACT",
    "ACCOUNT", "DELIVERY", "SUBSCRIPTION"
]

intents = {
    "ORDER": ["cancel_order", "change_order", "place_order", "track_order"],
    "SHIPPING": ["change_shipping_address", "set_up_shipping_address"],
    "CANCEL": ["check_cancellation_fee"],
    "INVOICE": ["check_invoice", "get_invoice"],
    "PAYMENT": ["check_payment_methods", "payment_issue"],
    "REFUND": ["check_refund_policy", "get_refund", "track_refund"],
    "FEEDBACK": ["complaint", "review"],
    "CONTACT": ["contact_customer_service", "contact_human_agent"],
    "ACCOUNT": ["create_account", "delete_account", "edit_account", "recover_password", "registration_problems", "switch_account"],
    "DELIVERY": ["delivery_options", "delivery_period"],
    "SUBSCRIPTION": ["newsletter_subscription"]
}